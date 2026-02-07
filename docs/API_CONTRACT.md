# API Contract

## Layered Service

Base URL default: `http://127.0.0.1:8101`

### `GET /healthz`

Response:

```json
{"status": "ok"}
```

### `GET /readyz`

Response:

```json
{"ready": true, "backend": "qwen", "last_error": null}
```

### `POST /infer`

Request fields:

- `request_id` (string, required)
- `sample_id` (string, optional)
- `image_path` (string, optional)
- `image_b64` (string, optional, PNG base64)
- `return_b64` (bool, default `true`)
- `save_cache` (bool, default `true`)

Response fields:

- `request_id`
- `runtime`
- `width`, `height`
- `layers[]`
  - `layer_id`
  - `rgba_b64` / `rgba_path`
  - `alpha_b64` / `alpha_path`
- `cache_dir`

## Edit Service

Base URL default: `http://127.0.0.1:8102`

### `GET /healthz`

Response:

```json
{"status": "ok"}
```

### `GET /readyz`

Response:

```json
{"ready": true, "backend": "qwen", "last_error": null}
```

### `POST /infer`

Request fields:

- `request_id` (string, required)
- `sample_id` (string, optional)
- `image_path` (string, optional)
- `image_b64` (string, optional)
- `mask_path` (string, optional)
- `mask_b64` (string, optional)
- `prompt` (string, optional)
- `return_b64` (bool, default `true`)
- `save_cache` (bool, default `true`)

Response fields:

- `request_id`
- `runtime`
- `width`, `height`
- `result_image_b64` / `result_image_path`
- `cache_dir`

## Error codes

- `429 queue_full`: queue is full
- `504 infer_timeout`: inference timeout
- `422 invalid_input`: missing required input
- `500 infer_failed`: backend runtime failed
