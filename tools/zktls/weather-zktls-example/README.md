# Weather zkTLS Example

Use zkTLS to prove that weather data originates from the Open-Meteo API.

## Overview

zkTLS (Zero-Knowledge TLS) can generate cryptographic proofs — without disclosing the full response — that verify:
1. The data genuinely originates from the specified HTTPS endpoint
2. The data was not tampered with in transit
3. The data was obtained at a specific time

This example only generates an attestation over the Weather API response, with coordinates supplied at runtime.

## Usage

### 1. Install dependencies

```bash
npm install
```

### 2. Configure 

Copy `.env.example` to `.env` and fill in your `appId` and `appSecret`:

```bash
cp .env.example .env
```

### 3. Run

```bash
# Defaults to Shanghai coordinates
npm start
```
 
## API Reference

This example uses the free [Open-Meteo](https://open-meteo.com/) Weather API:

- **Weather API**: `https://api.open-meteo.com/v1/forecast`


