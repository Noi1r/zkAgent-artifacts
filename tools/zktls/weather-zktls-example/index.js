import { PrimusCoreTLS } from "@primuslabs/zktls-core-sdk";
import { configDotenv } from "dotenv";

configDotenv();


const appId = process.env.APP_ID || "0xd5a39048d42a3373a4d69f4b2d90d6e39a108eeb";
const appSecret = process.env.APP_SECRET || "0xdfcdf3ed38bf778ffbe9fb497552aaacc7070c735503757d6a0ec51e8712a112";

const DEFAULT_CITY = "Shanghai";
const DEFAULT_LATITUDE = 31.2222;
const DEFAULT_LONGITUDE = 121.4581;

async function proveWeatherData(city, latitude, longitude) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`Generating weather data attestation for "${city}"`);
    console.log(`${"=".repeat(60)}\n`);

    const zkTLS = new PrimusCoreTLS();
    const initResult = await zkTLS.init(appId, appSecret);
    console.log("zkTLS SDK initialization:", initResult ? "success" : "failed");

    console.log(`\nCity coordinates: latitude=${latitude}, longitude=${longitude}`);

    const weatherResult = await getWeatherWithProof(zkTLS, latitude, longitude, city);

    return weatherResult;
}

async function getWeatherWithProof(zkTLS, latitude, longitude, city) {
    console.log(`\n--- Step 1: Fetch weather data ---`);

    const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true&temperature_unit=celsius`;

    const request = {
        url: weatherUrl,
        method: "GET",
        header: {
            "Accept": "application/json"
        },
        body: ""
    };

    const responseResolves = [
        {
            keyName: 'temperature',
            parsePath: '$.current_weather.temperature',
        },
        {
            keyName: 'windspeed',
            parsePath: '$.current_weather.windspeed',
        },
        {
            keyName: 'weathercode',
            parsePath: '$.current_weather.weathercode',
        },
        {
            keyName: 'time',
            parsePath: '$.current_weather.time',
        }
    ];

    console.log("Request URL:", weatherUrl);

    const generateRequest = zkTLS.generateRequestParams(request, responseResolves);

    generateRequest.setAttMode({
        algorithmType: "proxytls"
    });

    console.log("Generating weather data attestation...");
    const weatherProveStart = Date.now();
    const attestation = await zkTLS.startAttestation(generateRequest);
    const weatherProveMs = Date.now() - weatherProveStart;
    console.log(`Weather attestation generation time: ${weatherProveMs}ms`);

    const weatherVerifyStart = Date.now();
    const verifyResult = zkTLS.verifyAttestation(attestation);
    const weatherVerifyMs = Date.now() - weatherVerifyStart;
    console.log("Weather attestation verification result:", verifyResult ? "✓ passed" : "✗ failed");
    console.log(`Weather attestation verification time: ${weatherVerifyMs}ms`);

    console.log("\n--- Attestation details ---");
    const attestationJson = JSON.stringify(attestation, null, 2);
    console.log("Attestation data:", attestationJson);
    if (attestation && attestation.data) {
        const data = typeof attestation.data === 'string' ? JSON.parse(attestation.data) : attestation.data;
        if (data && data.time) {
            console.log(`Weather data timestamp: ${data.time}`);
        }
    }

    const proofSizeBytes = Buffer.byteLength(JSON.stringify(attestation), "utf8");

    return {
        city: city,
        coordinates: { latitude, longitude },
        attestation: attestation,
        verified: verifyResult,
        proveTimeMs: weatherProveMs,
        verifyTimeMs: weatherVerifyMs,
        proofSizeBytes: proofSizeBytes
    };
}

async function main() {
    try {
        const argLatitude = process.argv[2];
        const argLongitude = process.argv[3];
        const argCity = process.argv[4];

        if ((argLatitude && !argLongitude) || (!argLatitude && argLongitude)) {
            throw new Error('Usage: node index.js <latitude> <longitude> [city]');
        }

        const latitude = argLatitude ? Number(argLatitude) : DEFAULT_LATITUDE;
        const longitude = argLongitude ? Number(argLongitude) : DEFAULT_LONGITUDE;
        const city = argLatitude ? (argCity || "Custom") : DEFAULT_CITY;

        if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
            throw new Error("latitude/longitude must be numbers");
        }

        const result = await proveWeatherData(city, latitude, longitude);

        console.log(`\n${"=".repeat(60)}`);
        console.log("Attestation generation complete!");
        console.log(`${"=".repeat(60)}`);
        console.log(`City: ${result.city}`);
        console.log(`Coordinates: (${result.coordinates.latitude}, ${result.coordinates.longitude})`);
        console.log(`Attestation verification: ${result.verified ? "✓ valid" : "✗ invalid"}`);
        console.log(`Prove time:  ${result.proveTimeMs} ms`);
        console.log(`Verify time: ${result.verifyTimeMs} ms`);
        console.log(`Proof size:  ${result.proofSizeBytes} bytes (${(result.proofSizeBytes / 1024).toFixed(2)} KB)`);
        console.log(`\nThis attestation can prove to third parties that the weather data truly originates from the Open-Meteo API`);

    } catch (error) {
        console.error("Error:", error.message);
        process.exit(1);
    }

    process.exit(0);
}

main();
