"""Прочитать ЗАГОЛОВОК safetensors и прогнать по нему конвертер.

В safetensors первые 8 байт — длина JSON-заголовка, а в самом заголовке лежат имена
всех тензоров с формами и типами. Этого достаточно, чтобы проверить маппинг: качать
гигабайты весов не нужно.

    python inspect_checkpoint.py /path/to/model.safetensors
    python inspect_checkpoint.py https://civitai.com/api/download/models/3094215

Для закрытых ссылок CivitAI нужен токен в CIVITAI_TOKEN (скрипт его только
подставляет в заголовок запроса и нигде не печатает).
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

HEADER_PROBE = 2 * 1024 * 1024        # первых мегабайтов хватает на заголовок с запасом


def read_header_local(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


class _DropAuthOnRedirect(__import__("urllib.request", fromlist=["x"]).HTTPRedirectHandler):
    """CivitAI редиректит на CDN с подписанным URL.

    Заголовок Authorization на чужом хосте не нужен и ломает подпись: CDN отвечает 403.
    Снимаем его при смене хоста — ровно так поступает curl -L.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        import urllib.parse
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlparse(req.full_url).netloc
            if urllib.parse.urlparse(newurl).netloc != old_host:
                for h in ("Authorization", "authorization"):
                    new.headers.pop(h, None)
        return new


def read_header_remote(url: str) -> dict:
    import urllib.error
    import urllib.request

    token = os.environ.get("CIVITAI_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ВНИМАНИЕ: CIVITAI_TOKEN не задан в этой оболочке — закрытые файлы не отдадутся")

    opener = urllib.request.build_opener(_DropAuthOnRedirect())

    def attempt(use_range: bool):
        headers = {"User-Agent": "krea2-studio/1.0"}
        if use_range:
            headers["Range"] = f"bytes=0-{HEADER_PROBE - 1}"
        req = urllib.request.Request(url, headers=headers)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with opener.open(req, timeout=120) as r:
            # Без Range сервер начнёт отдавать все 12 ГБ — читаем начало и обрываем.
            return r.read(HEADER_PROBE)

    try:
        blob = attempt(use_range=True)
    except urllib.error.HTTPError as e:
        body = e.read()[:400].decode("utf-8", "replace")
        if e.code in (401, 403):
            raise RuntimeError(
                f"HTTP {e.code} от CivitAI. Ответ сервера:\n  {body}\n\n"
                "Частые причины:\n"
                "  * в аккаунте не включён показ Mature/X — файл помечен nsfw,\n"
                "    настройки: civitai.com/user/account -> Content Moderation;\n"
                "  * токен отозван или скопирован не полностью;\n"
                "  * автор ограничил скачивание (подписка, ранний доступ).\n"
                "Проверь, что файл вообще доступен: открой страницу модели в браузере\n"
                "под тем же аккаунтом и нажми Download."
            ) from e
        if e.code in (416, 501):                      # Range не поддержан
            print(f"  Range отклонён (HTTP {e.code}), пробую обычным запросом")
            blob = attempt(use_range=False)
        else:
            raise RuntimeError(f"HTTP {e.code}: {body}") from e

    if blob[:1] == b"{":                                  # вместо весов приехал JSON с ошибкой
        raise RuntimeError(f"сервер вернул не файл: {blob[:200].decode('utf-8', 'replace')}")
    (n,) = struct.unpack("<Q", blob[:8])
    if n + 8 > len(blob):
        raise RuntimeError(f"заголовок {n/1024**2:.1f} МБ не влез в пробу — увеличь HEADER_PROBE")
    return json.loads(blob[8:8 + n])


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]

    print(f"источник: {src}")

    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("checkpoint_header.json")
    if src.startswith(("http://", "https://")):
        if cache.exists():
            print(f"беру заголовок из кэша: {cache}")
            header = json.loads(cache.read_text())
        else:
            header = read_header_remote(src)
            header.pop("__metadata__", None)
            cache.write_text(json.dumps(header))
            print(f"заголовок сохранён в {cache} — повторно качать не нужно")
    else:
        header = read_header_local(Path(src))
    header.pop("__metadata__", None)

    print(f"тензоров в файле: {len(header)}")
    dtypes: dict[str, int] = {}
    total = 0
    for info in header.values():
        dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + 1
        a, b = info["data_offsets"]
        total += b - a
    print(f"типы: {dtypes}")
    print(f"вес данных: {total/1024**3:.2f} ГБ")

    print("\nпервые 8 имён как есть:")
    for k in list(header)[:8]:
        print(f"  {k:<52} {header[k]['dtype']:<10} {header[k]['shape']}")

    # Прогоняем конвертер по именам: вместо весов кладём пустышки нужной формы.
    sys.path.insert(0, str(Path(__file__).parent))
    import torch
    from krea2_studio.checkpoint import convert_comfy_state_dict, infer_config

    # device="meta" — тензор существует только как форма, память не выделяется.
    # Иначе 430 тензоров этой модели развернулись бы в ~48 ГБ fp32 и убили процесс.
    fake = {k: torch.zeros(v["shape"] or [1], device="meta") for k, v in header.items()}
    print(f"\n--- конвертация {len(fake)} имён ---")
    try:
        out = convert_comfy_state_dict(fake, strict=True)
    except ValueError as e:
        print(f"НЕ СОШЛОСЬ:\n{e}")
        return 1

    print(f"сопоставлено: {len(out)} из {len(fake)}")
    try:
        cfg = infer_config(out)
        print("\nконфиг, восстановленный из форм:")
        for k, v in cfg.items():
            print(f"  {k:<24} {v}")
        hidden = cfg["attention_head_dim"] * cfg["num_attention_heads"]
        print(f"  {'(hidden_size)':<24} {hidden}")
        ok = hidden == 6144 and cfg["num_layers"] == 28 and cfg["text_hidden_dim"] == 2560
        print(f"\nсовпадает со штатным Krea 2 (28 слоёв, hidden 6144, text 2560): {ok}")
    except Exception as e:
        print(f"конфиг восстановить не удалось: {type(e).__name__}: {e}")
        return 1

    print("\nмаппинг работает на этом файле")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
