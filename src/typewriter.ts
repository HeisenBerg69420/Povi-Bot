export interface TypewriterOptions {
  delayMs?: number;
  signal?: AbortSignal;
}

export interface TypewriterTarget {
  textContent: string | null;
}

export async function typeText(
  element: TypewriterTarget,
  text: string,
  options: TypewriterOptions = {},
): Promise<void> {
  const delayMs = Math.max(0, options.delayMs ?? 45);
  element.textContent = "";

  for (const character of text) {
    if (options.signal?.aborted) {
      return;
    }

    element.textContent += character;

    if (delayMs > 0) {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, delayMs);
      });
    }
  }
}
