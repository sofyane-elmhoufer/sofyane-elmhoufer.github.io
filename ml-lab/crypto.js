// Déchiffrement du contenu : PBKDF2-SHA256 -> AES-256-GCM (WebCrypto).
// Le mot de passe n'est jamais stocké : il sert uniquement à dériver la clé dans le navigateur.
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const AAD = "ml-lab-v1";

export async function decryptPayload(env, password) {
  const enc = new TextEncoder();
  const subtle = globalThis.crypto.subtle;
  const keyMaterial = await subtle.importKey("raw", enc.encode(password), "PBKDF2", false, ["deriveKey"]);
  const key = await subtle.deriveKey(
    { name: "PBKDF2", salt: b64(env.salt), iterations: env.iter, hash: "SHA-256" },
    keyMaterial,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"]
  );
  // Lève une erreur si le mot de passe est faux ou si le fichier a été modifié (authentification GCM).
  const plain = await subtle.decrypt(
    { name: "AES-GCM", iv: b64(env.iv), additionalData: enc.encode(AAD) },
    key,
    b64(env.data)
  );
  return JSON.parse(new TextDecoder().decode(plain));
}
