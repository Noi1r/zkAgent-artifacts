'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { createHash, randomUUID } = require('node:crypto');

// Backport only the official production signer rotation; keep SDK verification intact.
// https://github.com/primus-labs/zktls-core-sdk/commit/a184af6c646e509453616cad7c4069bbe8e5bef2
const SDK_NAME = '@primuslabs/zktls-core-sdk';
const SDK_VERSION = '0.2.11';
const OLD_SIGNER = '0xDB736B13E2f522dBE18B2015d0291E4b193D8eF6';
const PRODUCTION_SIGNER = '0x5Ed3176045F68617B0958a61ac1868875C98C14d';
const ORIGINAL_SHA256 = '4b09a6ddb3ff9114c564e500d9fc28aab3ff5b4e7e053adb1e6f916f755e8b24';
const PATCHED_SHA256 = '2708ecf451dbcb3365461a1c461201b26401e9d4fd470a3eae51fa3ac0485098';

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function patchSdkSigner(sdkDir = path.join(__dirname, 'node_modules', SDK_NAME)) {
  const metadata = JSON.parse(fs.readFileSync(path.join(sdkDir, 'package.json'), 'utf8'));
  if (metadata.name !== SDK_NAME || metadata.version !== SDK_VERSION) {
    throw new Error(`Signer backport requires exactly ${SDK_NAME}@${SDK_VERSION}`);
  }

  const constantsPath = path.join(sdkDir, 'dist', 'config', 'constants.js');
  const original = fs.readFileSync(constantsPath);
  const originalHash = sha256(original);
  if (originalHash === PATCHED_SHA256) return 'already-patched';
  if (originalHash !== ORIGINAL_SHA256) {
    throw new Error('Unexpected SDK constants.js SHA-256; refusing signer backport');
  }

  const patched = Buffer.from(original.toString('utf8').replace(OLD_SIGNER, PRODUCTION_SIGNER));
  if (sha256(patched) !== PATCHED_SHA256) {
    throw new Error('Unexpected patched SDK constants.js SHA-256');
  }

  const temporaryPath = `${constantsPath}.${randomUUID()}.tmp`;
  try {
    fs.writeFileSync(temporaryPath, patched, {
      flag: 'wx',
      mode: fs.statSync(constantsPath).mode & 0o777,
    });
    fs.renameSync(temporaryPath, constantsPath);
  } finally {
    fs.rmSync(temporaryPath, { force: true });
  }
  return 'patched';
}

module.exports = { patchSdkSigner };

if (require.main === module) {
  try {
    console.log(`SDK 0.2.11 official production signer backport: ${patchSdkSigner()}`);
  } catch (error) {
    console.error(`SDK signer backport failed: ${error.message}`);
    process.exitCode = 1;
  }
}
