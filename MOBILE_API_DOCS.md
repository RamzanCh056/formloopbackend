# FormLoop Mobile API Documentation

Complete reference for integrating the FormLoop API into a mobile app.

**Base URL:** `https://formloop.app`  
**Auth:** Session cookie (`session` header) — set automatically after login  
**Content-Type:** `multipart/form-data` for file uploads, `application/json` for save requests

---

## Authentication

FormLoop uses server-side sessions via a signed cookie named `session`.  
After your existing login flow calls `/auth/session`, the server sets this cookie.  
**Include the cookie on every subsequent request.** Most HTTP clients (Dio, Alamofire, Retrofit) handle this automatically with a cookie jar / credential store.

```
Cookie: session=<signed-session-value>
```

Requests without a valid session cookie are treated as guest (anonymous) — processing still works but saving to the library requires authentication.

---

## 1. Health Check

Verify the server is online and models are loaded.

**`GET /health`**

No headers or params required.

### Response `200 OK`

```json
{
  "ok": true,
  "has_checkpoint": true,
  "rvm_ready": true,
  "pro_ready": true,
  "outputs_dir": "/app/api_outputs",
  "save_flow_version": "2"
}
```

| Field | Type | Description |
|---|---|---|
| `ok` | bool | `true` when at least one model is ready |
| `rvm_ready` | bool | Classic RVM model available |
| `pro_ready` | bool | Pro (BiRefNet) model available — used on Railway |
| `save_flow_version` | string | Internal version marker |

---

## 2. GIF Processing — Full Flow

The recommended flow for mobile is async (non-blocking):

```
POST /api/v1/matte/start   →  get job_id
  ↓ poll every 2-3 seconds
GET /api/v1/matte/progress/{job_id}   →  until status == "completed"
  ↓ read result.gif_url / result.webm_url from progress response
GET /api/v1/matte/files/{job_id}/matte.gif   →  download GIF
POST /api/v1/matte/save/{job_id}   →  save to Firebase library (optional)
```

---

### 2a. Start Processing

**`POST /api/v1/matte/start`**

Upload a video and start background removal. Returns immediately with a `job_id` — processing happens asynchronously.

#### Headers

```
Cookie: session=<value>
Content-Type: multipart/form-data
```

#### Request — multipart/form-data

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | ✅ | — | Video file (MP4, MOV, MKV, WebM, M4V) |

#### Query Parameters

| Param | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `gif_width` | int | `960` | 320–1280 | Output GIF width in pixels. Height is auto-scaled. |
| `gif_fps` | int | `0` | 0–30 | Frames per second. `0` = auto-detect from source (capped at 24). |
| `loop_style` | string | `"normal"` | `normal`, `reverse` | `normal` = loops forward. `reverse` = boomerang (forward then backward seamlessly). |
| `rotation` | int | `0` | 0, 90, 180, 270 | Clockwise rotation in degrees before processing. |
| `start_time` | float | `null` | seconds | Trim start time. Omit to use beginning of clip. |
| `end_time` | float | `null` | seconds | Trim end time. Omit to use full clip. |
| `gif_white_bg` | bool | `false` | — | `true` = opaque white background GIF. `false` = transparent GIF (recommended). |
| `transparent_formats` | string | `"gif"` | `gif`, `webm` | Comma-separated formats. `webm` also generates a transparent WebM. |
| `model` | string | `"pro"` | `rvm`, `rembg`, `pro` | Always `pro` on Railway (server overrides automatically). |
| `premium` | bool | `false` | — | `true` = higher-quality pipeline (slower). |
| `fast_mode` | bool | `true` | — | BiRefNet fast mode. Leave `true` for mobile. |

#### Recommended Parameters for Mobile

```
gif_width=960
gif_fps=0
loop_style=normal        (or reverse for boomerang)
transparent_formats=gif
model=pro
premium=false
gif_white_bg=false
```

#### Response `200 OK`

```json
{
  "success": true,
  "job_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
  "status": "queued",
  "progress": 0,
  "message": "RunPod processing started",
  "progress_url": "https://formloop.app/api/v1/matte/progress/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string | 32-char hex ID — store this, used in all follow-up calls |
| `status` | string | Always `"queued"` on start |
| `progress_url` | string | Convenience — full URL to poll for progress |

#### Error Responses

| Status | Meaning |
|---|---|
| `400` | File too small / empty |
| `403` | Monthly GIF quota reached — user must upgrade |
| `413` | File too large (default max 300 MB) |
| `503` | Server not ready (model loading or RunPod not configured) |

#### File Limits

- **Max size:** 300 MB (default, set by `RVM_MAX_UPLOAD_MB` env var on server)
- **Supported formats:** `.mp4`, `.mov`, `.mkv`, `.webm`, `.m4v`
- **Recommended clip length:** 1–10 seconds for best results and speed

---

### 2b. Poll for Progress

**`GET /api/v1/matte/progress/{job_id}`**

Poll this endpoint every 2–3 seconds until `done` is `true`.

#### Path Params

| Param | Description |
|---|---|
| `job_id` | 32-char hex from the start response |

#### Response — while processing

```json
{
  "job_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
  "status": "running",
  "progress": 45,
  "message": "Encoding transparent GIF",
  "done": false
}
```

#### Response — on completion

```json
{
  "job_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
  "status": "completed",
  "progress": 100,
  "message": "Completed",
  "done": true,
  "result": {
    "success": true,
    "job_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
    "gif_url": "https://formloop.app/api/v1/matte/files/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4/matte.gif",
    "webm_url": "https://formloop.app/api/v1/matte/files/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4/matte_transparent.webm",
    "gif_width": 960
  }
}
```

#### Response — on failure

```json
{
  "job_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
  "status": "failed",
  "progress": 100,
  "message": "Processing failed",
  "done": true,
  "error": "Processing failed (subprocess exit 1)"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `queued` → `running` → `completed` or `failed` |
| `progress` | int | 0–100 |
| `done` | bool | `true` when finished (either completed or failed) |
| `result.gif_url` | string | Direct URL to transparent GIF |
| `result.webm_url` | string | Direct URL to transparent WebM (if requested) |

#### Error Responses

| Status | Meaning |
|---|---|
| `400` | Invalid job_id format |
| `404` | Job not found (expired or never existed) |

---

### 2c. Download Result File

**`GET /api/v1/matte/files/{job_id}/{filename}`**

Download any output file by name. Use the URLs from the progress `result` object directly — they point here.

#### Path Params

| Param | Description |
|---|---|
| `job_id` | 32-char hex job ID |
| `filename` | One of the allowed filenames below |

#### Allowed Filenames

| Filename | Format | Description |
|---|---|---|
| `matte.gif` | GIF | Transparent animated GIF ← **main output** |
| `matte_transparent.webm` | WebM | Transparent WebM video (VP9 alpha) |
| `foreground.mp4` | MP4 | Foreground (subject) video |
| `alpha.mp4` | MP4 | Alpha matte video |

#### Response

Returns the file as a binary stream with appropriate `Content-Type`.

| Status | Meaning |
|---|---|
| `200` | File returned |
| `404` | Job or file not found |

---

### 2d. Save Export to Library

**`POST /api/v1/matte/save/{job_id}`**

Save a processed GIF to the user's Firebase library. Requires authentication.

#### Headers

```
Cookie: session=<value>
Content-Type: application/json
```

#### Path Params

| Param | Description |
|---|---|
| `job_id` | 32-char hex job ID |

#### Request Body (JSON)

```json
{
  "export_id": "unique-export-identifier",
  "gif_url": "https://formloop.app/api/v1/matte/files/a1b2c3d4.../matte.gif",
  "webm_url": "https://formloop.app/api/v1/matte/files/a1b2c3d4.../matte_transparent.webm",
  "platform": "powerpoint"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `export_id` | string | ✅ | Unique ID for this export (1–120 chars, alphanumeric + `_-`). Generate with UUID on client. |
| `gif_url` | string | ✅ | The `gif_url` from the progress result |
| `webm_url` | string | ❌ | The `webm_url` from the progress result (optional) |
| `platform` | string | ❌ | Target platform — see Platform Guide below |

#### Platform Values

| Value | Platform | Output format |
|---|---|---|
| `powerpoint` | Microsoft PowerPoint | WebM (transparent) |
| `google-slides` | Google Slides | WebM (transparent) |
| `keynote` | Apple Keynote | WebM (transparent) |
| `canva` | Canva | GIF |
| `other` | Other / General | Both |

#### Response `200 OK`

```json
{
  "ok": true,
  "job_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
  "storageGifUrl": "https://firebasestorage.googleapis.com/v0/b/your-bucket/o/users%2F.../matte.gif?alt=media&token=...",
  "storageWebmUrl": "https://firebasestorage.googleapis.com/v0/b/your-bucket/o/users%2F.../matte_transparent.webm?alt=media&token=..."
}
```

| Field | Type | Description |
|---|---|---|
| `ok` | bool | `true` on success |
| `storageGifUrl` | string | Permanent Firebase Storage URL for the GIF |
| `storageWebmUrl` | string | Permanent Firebase Storage URL for the WebM (null if not saved) |

#### Error Responses

| Status | Meaning |
|---|---|
| `400` | Invalid job_id |
| `401` | Not signed in |
| `403` | Job belongs to a different user |
| `404` | Job not found and no gif_url provided |

---

### 2e. Download via Proxy (Firebase URLs)

**`GET /api/v1/matte/download`**

Proxy-download a file from Firebase Storage through the FormLoop server. Use this when you need the server to fetch a Firebase URL on behalf of the client.

#### Query Parameters

| Param | Type | Required | Description |
|---|---|---|---|
| `url` | string | ✅ | Must be an `https://` Firebase Storage URL |
| `filename` | string | ❌ | Suggested download filename (default: `download.bin`) |

Only Firebase Storage URLs (`firebasestorage.googleapis.com`, `storage.googleapis.com`) are allowed.

#### Response

Binary file with `Content-Disposition: attachment` header.

| Status | Meaning |
|---|---|
| `200` | File returned |
| `400` | URL not from an allowed host |
| `502` | Upstream fetch failed |

---

## 3. GIF Library

### 3a. Get User's Saved GIFs

**`GET /dashboard/gifs`**

Returns an HTML page. For a mobile app, use the Firebase Firestore SDK directly to read the user's exports from:

```
Firestore path: users/{uid}/exports
```

Each export document contains:

```json
{
  "jobId": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
  "gifUrl": "https://firebasestorage.googleapis.com/...",
  "webmUrl": "https://firebasestorage.googleapis.com/...",
  "title": "My Workout",
  "platform": "powerpoint",
  "createdAt": "2025-05-25T12:00:00Z",
  "customTags": ["legs", "workout"]
}
```

> **Note:** The `/dashboard/gifs` endpoint returns HTML (server-side rendered), not JSON. For mobile, read Firestore directly using the Firebase SDK with the user's auth token.

---

### 3b. Delete a Saved GIF

**`POST /dashboard/gifs/{job_id}/delete`**

Delete a GIF from the user's library (removes from Firestore and local storage).

#### Headers

```
Cookie: session=<value>
```

#### Path Params

| Param | Description |
|---|---|
| `job_id` | 32-char hex job ID to delete |

#### Response

`303 Redirect` to `/dashboard/gifs` on success.

For mobile, check that the redirect succeeds (final URL is `/dashboard/gifs`).

#### Error Responses

| Status | Meaning |
|---|---|
| `302` | Not signed in — redirects to login |
| `400` | Invalid job_id format |
| `403` | Job belongs to a different user |

> **Mobile alternative:** Delete directly from Firestore using the Firebase SDK:  
> Delete document at `users/{uid}/exports/{exportId}`.

---

## 4. User Profile

**`GET /profile`**

Returns an HTML page with user account details.

> This endpoint returns HTML. For mobile, read the user profile data from Firebase Auth and Firestore directly.

Key data available in Firestore:

```
users/{uid}/billing  →  plan_tier, stripe_customer_id, etc.
users/{uid}/exports  →  all saved GIF exports
```

#### Redirect Behavior

| Condition | Response |
|---|---|
| Not signed in | `302 → /auth/login?next=/profile` |
| Signed in | `200` HTML page |

---

## 5. Subscription

**`GET /subscription`**

Returns an HTML page showing available plans and upgrade options.

> Returns HTML — not suitable for direct mobile API consumption. Handle subscription upgrades via Stripe's hosted checkout (the page links to it) or use a deep link to open the subscription page in an in-app browser.

#### Available Plans

| Plan Key | Name | Features |
|---|---|---|
| `free` | Free | Limited GIFs per month, FormLoop watermark on exports |
| `starter` | Starter | More GIFs, no watermark, priority processing |
| `pro` | Pro | Highest limits, no watermark, all features |

---

## Complete Processing Flow Example

```
1. Upload video
   POST https://formloop.app/api/v1/matte/start
     ?gif_width=960&gif_fps=0&loop_style=reverse&transparent_formats=gif
   Body: multipart with file=<video.mp4>
   → { "job_id": "abc123...", "status": "queued" }

2. Poll until done (every 2-3 seconds)
   GET https://formloop.app/api/v1/matte/progress/abc123...
   → { "status": "running", "progress": 55, "done": false }
   → { "status": "running", "progress": 90, "done": false }
   → {
       "status": "completed",
       "done": true,
       "result": {
         "gif_url": "https://formloop.app/api/v1/matte/files/abc123.../matte.gif",
         "webm_url": "https://formloop.app/api/v1/matte/files/abc123.../matte_transparent.webm"
       }
     }

3. Download GIF (optional — use gif_url directly to display in app)
   GET https://formloop.app/api/v1/matte/files/abc123.../matte.gif

4. Save to library (requires login)
   POST https://formloop.app/api/v1/matte/save/abc123...
   Body: {
     "export_id": "550e8400-e29b-41d4-a716-446655440000",
     "gif_url": "https://formloop.app/api/v1/matte/files/abc123.../matte.gif",
     "webm_url": "https://formloop.app/api/v1/matte/files/abc123.../matte_transparent.webm",
     "platform": "canva"
   }
   → { "ok": true, "storageGifUrl": "https://firebasestorage.googleapis.com/..." }
```

---

## Processing Parameters Quick Reference

### loop_style

| Value | Behaviour | Use case |
|---|---|---|
| `normal` | Plays forward on repeat | Standard looping content |
| `reverse` | Forward then backward (boomerang) | Exercise demos, Instagram Reels style |

### gif_width

| Value | Quality | File size |
|---|---|---|
| `480` | Low | Smallest — fast preview |
| `640` | Medium | Good for mobile display |
| `960` | High ← **recommended** | Best quality/size balance |
| `1280` | Highest | Largest file size |

### rotation

Apply before processing so the model sees the correct orientation.

| Value | Effect |
|---|---|
| `0` | No rotation (default) |
| `90` | Rotate 90° clockwise |
| `180` | Rotate 180° (upside down) |
| `270` | Rotate 270° clockwise (= 90° counter-clockwise) |

### start_time / end_time

Trim the clip before processing — reduces processing time and file size.

```
POST /api/v1/matte/start?start_time=1.5&end_time=4.0
```

Processes only seconds 1.5 → 4.0 of the uploaded video.

### transparent_formats

| Value | Generates |
|---|---|
| `gif` | `matte.gif` only |
| `gif,webm` | `matte.gif` + `matte_transparent.webm` |

WebM (VP9 alpha) is the recommended format for PowerPoint, Google Slides, and Keynote.

---

## Platform Format Guide

| Platform | Best format | `platform` value | `transparent_formats` |
|---|---|---|---|
| PowerPoint | WebM | `powerpoint` | `gif,webm` |
| Google Slides | WebM | `google-slides` | `gif,webm` |
| Keynote | WebM | `keynote` | `gif,webm` |
| Canva | GIF | `canva` | `gif` |
| Other / General | Both | `other` | `gif,webm` |

---

## Error Code Summary

| HTTP Status | Meaning |
|---|---|
| `400` | Bad request — invalid params or file |
| `401` | Authentication required |
| `403` | Forbidden — quota exceeded or wrong user |
| `404` | Resource not found |
| `413` | File too large (max 300 MB) |
| `503` | Server not ready |
| `502` | Upstream error (RunPod or Firebase) |
