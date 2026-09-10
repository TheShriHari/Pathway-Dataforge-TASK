import markdown2
import subprocess
import os
import pymupdf
import re

def clean_latex(md_text):
    md_text = md_text.replace("$O(N^2)$", "<i>O</i>(<i>N</i>²)")
    md_text = md_text.replace("$$h_k = \\text{Core}(h_{k-1}, x) \\quad \\text{for } k \\in [1, K]$$",
        "<b><i>h<sub>k</sub></i> = Core(<i>h</i><sub><i>k</i>-1</sub>, <i>x</i>)</b> &nbsp;for <i>k</i> ∈ [1, <i>K</i>]")
    md_text = md_text.replace("$h_k = \\text{Core}(h_{k-1}, x)$",
        "<i>h<sub>k</sub></i> = Core(<i>h</i><sub><i>k</i>-1</sub>, <i>x</i>)")
    md_text = md_text.replace("($h \\in \\mathbb{R}^{C \\times H \\times W}$)",
        "(<i>h</i> ∈ ℝ<sup><i>C</i> × <i>H</i> × <i>W</i></sup>)")
    md_text = md_text.replace("($\\mathbb{R}^{48 \\times 15 \\times 15}$)", "(ℝ<sup>48 × 15 × 15</sup>)")
    md_text = md_text.replace("15 \\times 15", "15 × 15")
    md_text = md_text.replace("$15 \\times 15$", "15 × 15")
    md_text = md_text.replace("$K=1 \\to 10$", "<i>K</i> = 1 → 10")
    md_text = md_text.replace("$K=10 \\to 20$", "<i>K</i> = 10 → 20")
    md_text = md_text.replace("$K \\ge 11$", "<i>K</i> ≥ 11")
    md_text = md_text.replace("$K=1$", "<i>K</i> = 1")
    md_text = md_text.replace("$K=5$", "<i>K</i> = 5")
    md_text = md_text.replace("$K=10$", "<i>K</i> = 10")
    md_text = md_text.replace("$K=20$", "<i>K</i> = 20")
    md_text = md_text.replace("$K=4$", "<i>K</i> = 4")
    md_text = md_text.replace("$K=16–20$", "<i>K</i> = 16–20")
    md_text = md_text.replace("$K=16-20$", "<i>K</i> = 16–20")
    md_text = md_text.replace("$K \\sim \\text{Uniform}(1, 15)$", "<i>K</i> ~ Uniform(1, 15)")
    md_text = md_text.replace("$K$", "<i>K</i>")
    md_text = md_text.replace("$k$", "<i>k</i>")
    md_text = md_text.replace("$L$", "<i>L</i>")
    md_text = md_text.replace("$N$", "<i>N</i>")
    md_text = md_text.replace("+$7.0\\%$", "+7.0%")
    md_text = md_text.replace("$O(1)$", "<i>O</i>(1)")
    md_text = re.sub(r'\$([^\$]+)\$', r'\1', md_text)
    md_text = md_text.replace("\\times", "×")
    md_text = md_text.replace("\\to", "→")
    md_text = md_text.replace("\\ge", "≥")
    md_text = md_text.replace("\\%", "%")
    return md_text

def generate_one_page_summary_pdf(md_path, html_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_text = clean_latex(md_text)
    html_body = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks", "cuddled-lists"])

    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>One-Page Concept Summary - DataForge x Pathway</title>
<style>
  @page {{
    size: letter;
    margin: 0.42in 0.45in;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 8.8pt;
    line-height: 1.28;
    color: #0f172a;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 13.5pt;
    line-height: 1.15;
    margin: 0 0 3px 0;
    color: #0369a1;
    font-weight: 800;
  }}
  h3 {{
    font-size: 9.6pt;
    margin: 6px 0 2px 0;
    color: #0c4a6e;
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 1px;
    font-weight: 700;
    break-after: avoid;
  }}
  p {{
    margin: 0 0 4px 0;
  }}
  blockquote {{
    margin: 3px 0;
    padding: 4px 10px;
    background: #f0f9ff;
    border-left: 3px solid #0284c7;
    font-style: italic;
    color: #0369a1;
    font-size: 8.8pt;
    line-height: 1.3;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0;
    font-size: 8.2pt;
    line-height: 1.25;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 3px 6px;
    text-align: left;
  }}
  th {{
    background: #f8fafc;
    font-weight: 700;
    color: #0f172a;
  }}
  ul, ol {{
    margin: 0 0 4px 0;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 1.5px;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 4px 0;
  }}
  code {{
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 8.2pt;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 2px;
  }}
  a {{
    color: #0284c7;
    text-decoration: none;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)

    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={abs_pdf}",
        "--no-pdf-header-footer",
        abs_html
    ]
    subprocess.run(cmd, check=True)

    if os.path.exists(pdf_path):
        doc = pymupdf.open(pdf_path)
        print(f"SUCCESS: Generated {pdf_path} with {len(doc)} page(s). File size: {os.path.getsize(pdf_path):,} bytes.")
        return len(doc)
    else:
        print(f"ERROR: PDF was not generated for {pdf_path}.")
        return 0

def generate_blog_report_pdf(md_path, html_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_text = clean_latex(md_text)
    html_body = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks", "cuddled-lists"])

    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Thinking Without Talking - DataForge x Pathway Hackathon Blog Report</title>
<style>
  @page {{
    size: letter;
    margin: 0.65in 0.7in;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #111827;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 18pt;
    margin: 0 0 6px 0;
    color: #0f172a;
    font-weight: 700;
  }}
  h2 {{
    font-size: 13pt;
    margin: 20px 0 4px 0;
    color: #1e3a5f;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 3px;
    font-weight: 700;
  }}
  h3 {{
    font-size: 11pt;
    margin: 14px 0 3px 0;
    color: #1e293b;
    font-weight: 600;
  }}
  p {{
    margin: 0 0 8px 0;
  }}
  blockquote {{
    margin: 8px 0;
    padding: 8px 14px;
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    font-style: italic;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 9pt;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    text-align: left;
  }}
  th {{
    background: #f1f5f9;
    font-weight: 600;
  }}
  ul, ol {{
    margin: 0 0 8px 0;
    padding-left: 22px;
  }}
  li {{
    margin-bottom: 3px;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 12px 0;
  }}
  code {{
    font-family: monospace;
    font-size: 9pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 2px;
  }}
  pre {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 8.5pt;
    overflow-x: auto;
    white-space: pre-wrap;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)

    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={abs_pdf}",
        "--no-pdf-header-footer",
        abs_html
    ]
    subprocess.run(cmd, check=True)

    if os.path.exists(pdf_path):
        doc = pymupdf.open(pdf_path)
        print(f"SUCCESS: Generated {pdf_path} with {len(doc)} page(s). File size: {os.path.getsize(pdf_path):,} bytes.")
        return len(doc)
    else:
        print(f"ERROR: PDF was not generated for {pdf_path}.")
        return 0

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    # 1. Generate strictly 1-page summary PDF at comfortable readable font size
    generate_one_page_summary_pdf(
        md_path=os.path.join(base, "one-page-summary.md"),
        html_path=os.path.join(base, "one-page-summary.html"),
        pdf_path=os.path.join(base, "one-page-summary.pdf")
    )

    # 2. Generate multi-page blog report PDF
    generate_blog_report_pdf(
        md_path=os.path.join(base, "blog-report.md"),
        html_path=os.path.join(base, "blog-report.html"),
        pdf_path=os.path.join(base, "blog-report.pdf")
    )
