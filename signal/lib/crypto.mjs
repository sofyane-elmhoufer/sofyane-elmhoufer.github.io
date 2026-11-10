// Shared by build.mjs (local) and update.mjs (GitHub Action).
// PBKDF2-SHA256 -> AES-256-GCM, gzip before encrypt. The password is only ever read from an
// environment variable at build time and is never written to disk or into the repo.
import { webcrypto as wc } from 'node:crypto';
import zlib from 'node:zlib';

const subtle = wc.subtle;
export const ITER = 600000;
const enc = new TextEncoder();
const b64 = (u8) => Buffer.from(u8).toString('base64');
const unb64 = (s) => new Uint8Array(Buffer.from(s, 'base64'));

export const newSalt = () => b64(wc.getRandomValues(new Uint8Array(16)));

async function deriveKey(password, saltB64, iter) {
  const base = await subtle.importKey('raw', enc.encode(password), 'PBKDF2', false, ['deriveKey']);
  return subtle.deriveKey(
    { name: 'PBKDF2', salt: unb64(saltB64), iterations: iter, hash: 'SHA-256' },
    base,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

export async function encryptJSON(obj, password, aad, saltB64, iter = ITER) {
  const key = await deriveKey(password, saltB64, iter);
  const iv = wc.getRandomValues(new Uint8Array(12));
  const plain = zlib.gzipSync(Buffer.from(JSON.stringify(obj), 'utf8'), { level: 9 });
  const ct = new Uint8Array(await subtle.encrypt({ name: 'AES-GCM', iv, additionalData: enc.encode(aad) }, key, plain));
  return { v: 1, kdf: 'PBKDF2-SHA256', iter, salt: saltB64, iv: b64(iv), aad, gz: true, data: b64(ct) };
}

export async function decryptJSON(env, password, aad) {
  if (env.aad !== aad) throw new Error(`Unexpected payload label: ${env.aad}`);
  const key = await deriveKey(password, env.salt, env.iter);
  const plain = new Uint8Array(await subtle.decrypt({ name: 'AES-GCM', iv: unb64(env.iv), additionalData: enc.encode(aad) }, key, unb64(env.data)));
  return JSON.parse(zlib.gunzipSync(plain).toString('utf8'));
}

export const AAD_APP = 'signal-app-v1';
export const AAD_DATA = 'signal-data-v1';
