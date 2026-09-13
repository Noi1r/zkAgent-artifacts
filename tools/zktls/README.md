# zkTLS Weather Demo

This repo contains a single example project at `weather-zktls-example/` that shows how to generate and verify zkTLS attestations for the Open-Meteo Weather API.

## Usage

### 1) Install dependencies

```bash
cd weather-zktls-example
npm install
```

### 2) Configure API keys (recommended)

Get your `appId` and `appSecret` from the Primus developer portal:

- Sign up and create an app at: https://dev.primuslabs.xyz

Then copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

Edit `.env`:

```
APP_ID=0xyourAppId
APP_SECRET=0xyourAppSecret
```

### 3) Run

```bash
# Default: Shanghai coordinates
npm start
```

The script prints:

- Proof generation time and verification time
- Attestation payload (including the weather data timestamp)
- Verification result

## API Key Notes

- `weather-zktls-example/index.js` currently includes default test keys for demo purposes.
- Remove or replace these keys before uploading or publishing the repo.
- Do not commit your `.env` file; add it to ignore rules if needed.
