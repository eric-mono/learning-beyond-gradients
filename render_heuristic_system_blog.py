from pathlib import Path
import re

import markdown


ROOT = Path(__file__).resolve().parent
EN_MD = ROOT / "blog_heuristic_system.en.md"
ZH_MD = ROOT / "blog_heuristic_system.md"
HTML_PATH = ROOT / "blog_heuristic_system.html"
STYLE_SOURCE = ROOT / "blog_heuristic_policy_atari_mujoco.html"
PAGE_TITLE_EN = "Heuristic System: Software Evolves Through Metabolism"


LANG_SCRIPT = """(function () {
  const panes = {
    en: document.getElementById('article-en'),
    zh: document.getElementById('article-zh'),
  };
  const buttons = {
    en: document.getElementById('lang-en'),
    zh: document.getElementById('lang-zh'),
  };
  const titles = {
    en: 'Heuristic System: Software Evolves Through Metabolism',
    zh: 'Heuristic System：软件在代谢中进化',
  };
  const storageKey = 'heuristic_system_blog_lang_v1';

  function setLanguage(lang, updateUrl) {
    if (!panes[lang]) lang = 'en';
    for (const key of Object.keys(panes)) {
      const selected = key === lang;
      panes[key].hidden = !selected;
      buttons[key].classList.toggle('active', selected);
      buttons[key].setAttribute('aria-pressed', selected ? 'true' : 'false');
    }
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = titles[lang];
    try { window.localStorage.setItem(storageKey, lang); } catch (_) {}
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.hash = lang === 'zh' ? 'zh' : '';
      window.history.replaceState(null, '', url);
    }
  }

  buttons.en.addEventListener('click', () => setLanguage('en', true));
  buttons.zh.addEventListener('click', () => setLanguage('zh', true));

  let initial = 'en';
  if (window.location.hash === '#zh') {
    initial = 'zh';
  } else if (window.location.hash === '#en') {
    initial = 'en';
  } else {
    try {
      const saved = window.localStorage.getItem(storageKey);
      if (saved === 'en' || saved === 'zh') initial = saved;
    } catch (_) {}
  }
  setLanguage(initial, false);
})();
"""


def extract_style(html: str) -> str:
    match = re.search(r"<style>\n?(.*?)\n?  </style>", html, re.S)
    if match is None:
        raise RuntimeError(f"Could not find style block in {STYLE_SOURCE}")
    return match.group(1)


def render_markdown(path: Path) -> str:
    return markdown.markdown(
        path.read_text(),
        extensions=["extra", "fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )


def main() -> None:
    style = extract_style(STYLE_SOURCE.read_text())
    en_html = render_markdown(EN_MD)
    zh_html = render_markdown(ZH_MD)
    HTML_PATH.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PAGE_TITLE_EN}</title>
  <style>
{style}
  </style>
</head>
<body>
  <main class="page">
    <div class="lang-switch-wrap" aria-label="Language switcher">
      <div class="lang-switch">
        <button type="button" id="lang-en" class="active" aria-pressed="true">English</button>
        <button type="button" id="lang-zh" aria-pressed="false">中文</button>
      </div>
    </div>
    <article id="article-en" class="lang-pane" lang="en">
{en_html}
    </article>
    <article id="article-zh" class="lang-pane" lang="zh-CN" hidden>
{zh_html}
    </article>
  </main>
  <script>
{LANG_SCRIPT}
  </script>
</body>
</html>
"""
    )


if __name__ == "__main__":
    main()
