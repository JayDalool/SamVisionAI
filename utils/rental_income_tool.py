# utils/rental_income.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import urllib.parse
from io import BytesIO

import pandas as pd
import streamlit as st

# PDF generator (offline)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


SAMPLE_ROWS = [
    {
        "Property Name / Address": "123 Main St, Winnipeg",
        "Type (House / Condo)": "House",
        "Listing Price ($)": "399900",
        "Living Area (Sqft)": "1250",
        "Year Built": "1978",
        "Bedrooms": "3",
        "Bathrooms": "2",
        "Finished Basement (Yes/No)": "Yes",
        "Condo Fee ($/month) (0 if House)": "",
        "Reserve Fund (Yes/No + Notes if provided)": "",
        "Annual Property Tax ($/year)": "3200",
        "Annual Insurance ($/year)": "1800",
        "Monthly Rent ($/month)": "2400",
        "Mortgage ($/month) (optional)": "0",
    },
    {
        "Property Name / Address": "#2 74 Carlton St, Winnipeg",
        "Type (House / Condo)": "Condo",
        "Listing Price ($)": "214900",
        "Living Area (Sqft)": "820",
        "Year Built": "1988",
        "Bedrooms": "2",
        "Bathrooms": "1",
        "Finished Basement (Yes/No)": "No",
        "Condo Fee ($/month) (0 if House)": "480",
        "Reserve Fund (Yes/No + Notes if provided)": "Yes (healthy reserve fund)",
        "Annual Property Tax ($/year)": "2100",
        "Annual Insurance ($/year)": "900",
        "Monthly Rent ($/month)": "1950",
        "Mortgage ($/month) (optional)": "0",
    },
    {
        "Property Name / Address": "15 River Ave, Winnipeg",
        "Type (House / Condo)": "House",
        "Listing Price ($)": "489900",
        "Living Area (Sqft)": "1680",
        "Year Built": "2004",
        "Bedrooms": "4",
        "Bathrooms": "3",
        "Finished Basement (Yes/No)": "No",
        "Condo Fee ($/month) (0 if House)": "",
        "Reserve Fund (Yes/No + Notes if provided)": "",
        "Annual Property Tax ($/year)": "4200",
        "Annual Insurance ($/year)": "2100",
        "Monthly Rent ($/month)": "3100",
        "Mortgage ($/month) (optional)": "0",
    },
]

COLS = [
    "Property Name / Address",
    "Type (House / Condo)",
    "Listing Price ($)",
    "Living Area (Sqft)",
    "Price per Sqft ($/Sqft) (calculated)",
    "Year Built",
    "Bedrooms",
    "Bathrooms",
    "Finished Basement (Yes/No)",
    "Condo Fee ($/month) (0 if House)",
    "Reserve Fund (Yes/No + Notes if provided)",
    "Annual Property Tax ($/year)",
    "Annual Insurance ($/year)",
    "Monthly Rent ($/month)",
    "Mortgage ($/month) (optional)",
    "Monthly Expenses ($/month) (calculated)",
    "Net Monthly Income ($/month) (calculated)",
    "Net Annual Income ($/year) (calculated)",
]


def _to_num(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "").replace("$", ""))
    except Exception:
        return None


def _fmt_money_dash(n: Optional[float]) -> str:
    if n is None:
        return "—"
    return f"${n:,.2f}"


def _today_long() -> str:
    return datetime.now().strftime("%B %d, %Y")


def _safe_filename(name: str) -> str:
    s = (name or "Property").strip() or "Property"
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_()"
    out = "".join(ch if ch in allowed else "-" for ch in s)
    while "  " in out:
        out = out.replace("  ", " ")
    return (out.strip()[:80] or "Property")


def _escape_html(v: Any) -> str:
    s = "" if v is None else str(v)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def _calc_row(r: Dict[str, Any]) -> Dict[str, Optional[float]]:
    listing = _to_num(r.get("Listing Price ($)"))
    sqft = _to_num(r.get("Living Area (Sqft)"))
    tax = _to_num(r.get("Annual Property Tax ($/year)"))
    ins = _to_num(r.get("Annual Insurance ($/year)"))
    rent = _to_num(r.get("Monthly Rent ($/month)"))
    mort = _to_num(r.get("Mortgage ($/month) (optional)")) or 0.0

    typ = (r.get("Type (House / Condo)") or "House").strip()
    condo_fee_raw = _to_num(r.get("Condo Fee ($/month) (0 if House)"))

    pps = None
    if listing is not None and sqft not in (None, 0):
        pps = listing / float(sqft)

    if tax is None or ins is None:
        m_exp = None
    else:
        m_tax = tax / 12.0
        m_ins = ins / 12.0
        if typ.lower() == "condo":
            if condo_fee_raw is None:
                m_exp = None
            else:
                m_exp = m_tax + m_ins + float(condo_fee_raw) + mort
        else:
            m_exp = m_tax + m_ins + 0.0 + mort

    net_m = None
    net_a = None
    if rent is not None and m_exp is not None:
        net_m = rent - m_exp
        net_a = net_m * 12.0

    return {
        "pps": pps,
        "m_exp": m_exp,
        "net_m": net_m,
        "net_a": net_a,
        "m_tax": (tax / 12.0) if tax is not None else None,
        "m_ins": (ins / 12.0) if ins is not None else None,
        "rent": rent,
        "condo_fee": condo_fee_raw if typ.lower() == "condo" else 0.0,
        "mort": mort,
    }


def _build_summary_text(r: Dict[str, Any], c: Dict[str, Optional[float]]) -> str:
    # This is just for quick sharing; your main delivery will be PDF
    lines = []
    lines.append(f"Rental Income Report ({_today_long()})")
    lines.append(f"Property: {r.get('Property Name / Address') or '—'}")
    lines.append("")
    lines.append(f"Net Monthly: {_fmt_money_dash(c.get('net_m'))}")
    lines.append(f"Net Annual: {_fmt_money_dash(c.get('net_a'))}")
    lines.append(f"Monthly Rent: {_fmt_money_dash(c.get('rent'))}")
    lines.append(f"Monthly Expenses: {_fmt_money_dash(c.get('m_exp'))}")
    return "\n".join(lines)


def _build_report_html(r: Dict[str, Any], c: Dict[str, Optional[float]]) -> str:
    date = _escape_html(_today_long())
    addr = _escape_html(r.get("Property Name / Address") or "—")
    typ = _escape_html(r.get("Type (House / Condo)") or "—")

    year = _escape_html(r.get("Year Built") or "—")
    beds = _escape_html(r.get("Bedrooms") or "—")
    baths = _escape_html(r.get("Bathrooms") or "—")
    basement = _escape_html(r.get("Finished Basement (Yes/No)") or "—")

    listing = _fmt_money_dash(_to_num(r.get("Listing Price ($)")))
    sqft = _to_num(r.get("Living Area (Sqft)"))
    sqft_txt = "—" if sqft is None else f"{sqft:,.0f} sqft"

    pps = c.get("pps")
    pps_txt = "—" if pps is None else f"${pps:,.2f}"

    rent = _fmt_money_dash(c.get("rent"))
    m_tax = _fmt_money_dash(c.get("m_tax"))
    m_ins = _fmt_money_dash(c.get("m_ins"))
    condo_fee = _fmt_money_dash(c.get("condo_fee"))
    mort = _fmt_money_dash(c.get("mort") or 0.0)

    m_exp = _fmt_money_dash(c.get("m_exp"))
    net_m = _fmt_money_dash(c.get("net_m"))
    net_a = _fmt_money_dash(c.get("net_a"))

    reserve = _escape_html(r.get("Reserve Fund (Yes/No + Notes if provided)") or "—")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Rental Income Report - {addr}</title>
<style>
  :root{{--text:#0f172a;--muted:#64748b;--border:#e2e8f0;--bg:#ffffff;}}
  body{{font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:24px;}}
  .toolbar{{display:flex;justify-content:flex-end;gap:10px;margin-bottom:12px;}}
  .btn{{border:1px solid var(--border);background:#fff;border-radius:12px;padding:10px 14px;
       font-size:14px;font-weight:700;cursor:pointer;}}
  .btn:hover{{background:#f8fafc;}}
  .page{{max-width:860px;margin:0 auto;border:1px solid var(--border);border-radius:16px;padding:28px;}}
  .row{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;}}
  .title{{font-size:22px;font-weight:700;}}
  .muted{{color:var(--muted);font-size:12px;}}
  .strong{{font-weight:700;}}
  .cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:18px;}}
  .card{{border:1px solid var(--border);border-radius:14px;padding:14px;}}
  .cardLabel{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}}
  .cardValue{{font-size:20px;font-weight:800;margin-top:6px;}}
  .grid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px;}}
  .box{{border:1px solid var(--border);border-radius:14px;padding:14px;}}
  table{{width:100%;border-collapse:collapse;margin-top:10px;}}
  td{{padding:10px 0;border-bottom:1px solid var(--border);font-size:14px;}}
  td:last-child{{text-align:right;font-weight:600;}}
  .totals td{{border-bottom:none;}}
  .totals tr:first-child td{{border-top:1px solid var(--border);}}
  .note{{margin-top:16px;font-size:11px;color:var(--muted);}}

  @media print {{
    body{{padding:0;}}
    .page{{border:none;border-radius:0;padding:0;}}
    .no-print{{display:none !important;}}
    @page{{margin:16mm;}}
  }}
</style>
<script>
  function printReport(){{
    window.print();
  }}
</script>
</head>
<body>
  <div class="toolbar no-print">
    <button class="btn" onclick="printReport()">Print / Save PDF</button>
  </div>

  <div class="page">
    <div class="row">
      <div>
        <div class="title">Rental Income Report</div>
        <div class="muted">Date: {date}</div>
      </div>
      <div style="text-align:right">
        <div class="muted">Property</div>
        <div class="strong">{addr}</div>
      </div>
    </div>

    <div class="cards">
      <div class="card"><div class="cardLabel">Net Monthly</div><div class="cardValue">{net_m}</div></div>
      <div class="card"><div class="cardLabel">Net Annual</div><div class="cardValue">{net_a}</div></div>
      <div class="card"><div class="cardLabel">Monthly Rent</div><div class="cardValue">{rent}</div></div>
    </div>

    <div class="grid2">
      <div class="box">
        <div class="strong">Property Details</div>
        <table>
          <tr><td>Type</td><td>{typ}</td></tr>
          <tr><td>Listing Price</td><td>{listing}</td></tr>
          <tr><td>Living Area</td><td>{_escape_html(sqft_txt)}</td></tr>
          <tr><td>Price per Sqft</td><td>{pps_txt}</td></tr>
          <tr><td>Year Built</td><td>{year}</td></tr>
          <tr><td>Bedrooms</td><td>{beds}</td></tr>
          <tr><td>Bathrooms</td><td>{baths}</td></tr>
          <tr><td>Finished Basement</td><td>{basement}</td></tr>
        </table>
        {f'<div class="note"><span class="strong">Reserve Fund:</span> {reserve}</div>' if (r.get("Type (House / Condo)") or "").strip().lower()=="condo" else ''}
      </div>

      <div class="box">
        <div class="strong">Monthly Breakdown</div>
        <table>
          <tr><td>Income: Monthly Rent</td><td>{rent}</td></tr>
        </table>
        <div class="muted" style="margin-top:10px">Expenses</div>
        <table>
          <tr><td>Property Tax</td><td>{m_tax}</td></tr>
          <tr><td>Insurance</td><td>{m_ins}</td></tr>
          <tr><td>Condo Fee</td><td>{condo_fee}</td></tr>
          <tr><td>Mortgage</td><td>{mort}</td></tr>
        </table>
        <table class="totals">
          <tr><td class="strong">Total Monthly Expenses</td><td class="strong">{m_exp}</td></tr>
          <tr><td class="strong">Net Monthly Income</td><td class="strong">{net_m}</td></tr>
          <tr><td class="strong">Net Annual Income</td><td class="strong">{net_a}</td></tr>
        </table>
      </div>
    </div>

    <div class="note">
      Notes: Calculations stay blank until required values are entered. For condos, Condo Fee and Reserve Fund are required.
    </div>
  </div>
</body>
</html>"""


def _build_report_pdf_bytes(r: Dict[str, Any], c: Dict[str, Optional[float]]) -> bytes:
    """
    Offline, client-ready PDF. This is what you will email.
    """
    buf = BytesIO()
    w, h = letter
    left = 0.75 * inch
    right = w - 0.75 * inch
    y = h - 0.8 * inch
    line = 14

    def draw_label_value(label: str, value: str):
        nonlocal y
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left, y, label)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawRightString(right, y, value)
        y -= line

    pdf = canvas.Canvas(buf, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left, y, "Rental Income Report")
    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(right, y, f"Date: {_today_long()}")
    y -= 22

    address = str(r.get("Property Name / Address") or "—")
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(left, y, "Property:")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(left + 60, y, address[:80])
    y -= 18

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, y, f"Net Monthly: {_fmt_money_dash(c.get('net_m'))}")
    pdf.drawString(left + 200, y, f"Net Annual: {_fmt_money_dash(c.get('net_a'))}")
    pdf.drawString(left + 400, y, f"Monthly Rent: {_fmt_money_dash(c.get('rent'))}")
    y -= 18

    pdf.setLineWidth(0.6)
    pdf.line(left, y, right, y)
    y -= 18

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, y, "Property Details")
    y -= 16

    draw_label_value("Type", str(r.get("Type (House / Condo)") or "—"))
    draw_label_value("Listing Price", _fmt_money_dash(_to_num(r.get("Listing Price ($)"))))
    sqft = _to_num(r.get("Living Area (Sqft)"))
    draw_label_value("Living Area", "—" if sqft is None else f"{sqft:,.0f} sqft")
    pps = c.get("pps")
    draw_label_value("Price per Sqft", "—" if pps is None else f"${pps:,.2f}")
    draw_label_value("Year Built", str(r.get("Year Built") or "—"))
    draw_label_value("Bedrooms", str(r.get("Bedrooms") or "—"))
    draw_label_value("Bathrooms", str(r.get("Bathrooms") or "—"))
    draw_label_value("Finished Basement", str(r.get("Finished Basement (Yes/No)") or "—"))

    if str(r.get("Type (House / Condo)") or "").strip().lower() == "condo":
        draw_label_value("Reserve Fund", str(r.get("Reserve Fund (Yes/No + Notes if provided)") or "—"))

    y -= 8
    pdf.line(left, y, right, y)
    y -= 18

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, y, "Monthly Breakdown")
    y -= 16

    draw_label_value("Income: Monthly Rent", _fmt_money_dash(c.get("rent")))
    y -= 4
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(left, y, "Expenses")
    y -= 14

    draw_label_value("Property Tax", _fmt_money_dash(c.get("m_tax")))
    draw_label_value("Insurance", _fmt_money_dash(c.get("m_ins")))
    draw_label_value("Condo Fee", _fmt_money_dash(c.get("condo_fee")))
    draw_label_value("Mortgage", _fmt_money_dash(c.get("mort") or 0.0))

    y -= 6
    pdf.line(left, y, right, y)
    y -= 16

    pdf.setFont("Helvetica-Bold", 11)
    draw_label_value("Total Monthly Expenses", _fmt_money_dash(c.get("m_exp")))
    draw_label_value("Net Monthly Income", _fmt_money_dash(c.get("net_m")))
    draw_label_value("Net Annual Income", _fmt_money_dash(c.get("net_a")))

    y -= 10
    pdf.setFont("Helvetica", 8)
    pdf.setFillGray(0.4)
    pdf.drawString(left, y, "Note: For condos, Condo Fee and Reserve Fund are required for complete calculations.")
    pdf.setFillGray(0)

    pdf.showPage()
    pdf.save()
    return buf.getvalue()


def _open_link_button(label: str, url: str, key: str) -> None:
    """
    Real Streamlit button that opens a link in a new tab.
    """
    if st.button(label, key=key, use_container_width=True):
        st.components.v1.html(
            f"""
            <script>
              window.open("{url}", "_blank");
            </script>
            """,
            height=0,
        )


def render_rental_income_tab() -> None:
    st.header("🏠 Rental Income Analyzer")
    st.caption("DB-free calculator. Add multiple properties, compare net income, export a clean client report.")

    if "rental_rows" not in st.session_state:
        st.session_state["rental_rows"] = pd.DataFrame([{c: "" for c in COLS}])

    a, b, c, d = st.columns([1, 1, 1, 2])
    with a:
        if st.button("🌱 Load Sample Data", use_container_width=True):
            df_seed = pd.DataFrame(SAMPLE_ROWS)
            for col in COLS:
                if col not in df_seed.columns:
                    df_seed[col] = ""
            st.session_state["rental_rows"] = df_seed[COLS]
            st.rerun()

    with b:
        if st.button("➕ Add Row", use_container_width=True):
            st.session_state["rental_rows"] = pd.concat(
                [st.session_state["rental_rows"], pd.DataFrame([{c: "" for c in COLS}])],
                ignore_index=True,
            )
            st.rerun()

    with c:
        if st.button("🧹 Reset", use_container_width=True):
            st.session_state["rental_rows"] = pd.DataFrame([{c: "" for c in COLS}])
            st.rerun()

    with d:
        st.info("Condo rule: **Condo Fee** + **Reserve Fund** required for Condo. Mortgage is optional.")

    input_cols = [x for x in COLS if "(calculated)" not in x]
    edited = st.data_editor(
        st.session_state["rental_rows"][input_cols],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Type (House / Condo)": st.column_config.SelectboxColumn(
                "Type (House / Condo)", options=["House", "Condo"], required=True
            ),
            "Finished Basement (Yes/No)": st.column_config.SelectboxColumn(
                "Finished Basement (Yes/No)", options=["", "Yes", "No"]
            ),
            "Listing Price ($)": st.column_config.NumberColumn("Listing Price ($)", format="$%d"),
            "Annual Property Tax ($/year)": st.column_config.NumberColumn("Annual Property Tax ($/year)", format="$%d"),
            "Annual Insurance ($/year)": st.column_config.NumberColumn("Annual Insurance ($/year)", format="$%d"),
            "Monthly Rent ($/month)": st.column_config.NumberColumn("Monthly Rent ($/month)", format="$%d"),
            "Condo Fee ($/month) (0 if House)": st.column_config.NumberColumn("Condo Fee ($/month) (0 if House)", format="$%d"),
            "Mortgage ($/month) (optional)": st.column_config.NumberColumn("Mortgage ($/month) (optional)", format="$%d"),
            "Living Area (Sqft)": st.column_config.NumberColumn("Living Area (Sqft)", format="%d"),
            "Bedrooms": st.column_config.NumberColumn("Bedrooms", format="%d"),
            "Bathrooms": st.column_config.NumberColumn("Bathrooms", format="%.1f"),
            "Year Built": st.column_config.NumberColumn("Year Built", format="%d"),
        },
        key="rental_editor",
    )

    df_in = st.session_state["rental_rows"].copy()
    for col in input_cols:
        df_in[col] = edited[col]
    st.session_state["rental_rows"] = df_in

    computed_rows = []
    for _, row in df_in.iterrows():
        r = row.to_dict()
        ccalc = _calc_row(r)
        out = dict(r)
        out["Price per Sqft ($/Sqft) (calculated)"] = "" if ccalc["pps"] is None else round(ccalc["pps"], 2)
        out["Monthly Expenses ($/month) (calculated)"] = "" if ccalc["m_exp"] is None else round(ccalc["m_exp"], 2)
        out["Net Monthly Income ($/month) (calculated)"] = "" if ccalc["net_m"] is None else round(ccalc["net_m"], 2)
        out["Net Annual Income ($/year) (calculated)"] = "" if ccalc["net_a"] is None else round(ccalc["net_a"], 2)
        computed_rows.append(out)

    calc_df = pd.DataFrame(computed_rows, columns=COLS)

    st.subheader("📊 Results")
    st.dataframe(
        calc_df,
        use_container_width=True,
        column_config={
            "Price per Sqft ($/Sqft) (calculated)": st.column_config.NumberColumn(format="$%.2f"),
            "Monthly Expenses ($/month) (calculated)": st.column_config.NumberColumn(format="$%.2f"),
            "Net Monthly Income ($/month) (calculated)": st.column_config.NumberColumn(format="$%.2f"),
            "Net Annual Income ($/year) (calculated)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    net_m = pd.to_numeric(calc_df["Net Monthly Income ($/month) (calculated)"], errors="coerce")
    net_a = pd.to_numeric(calc_df["Net Annual Income ($/year) (calculated)"], errors="coerce")

    best_m_idx = int(net_m.idxmax()) if net_m.notna().any() else None
    best_a_idx = int(net_a.idxmax()) if net_a.notna().any() else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Net Monthly", calc_df.loc[best_m_idx, "Property Name / Address"] if best_m_idx is not None else "—")
    c2.metric("Best Net Annual", calc_df.loc[best_a_idx, "Property Name / Address"] if best_a_idx is not None else "—")
    c3.info("Driver: rent vs (tax + insurance + condo fee + mortgage).")

    st.divider()

    st.subheader("🧾 Client Report (PDF-first)")
    opts = []
    for i, v in enumerate(calc_df["Property Name / Address"].astype(str).tolist()):
        opts.append((i, v.strip() if v.strip() else f"Property {i+1}"))
    idx = st.selectbox("Select property", opts, format_func=lambda x: x[1])[0]

    selected = calc_df.iloc[idx].to_dict()
    ccalc = _calc_row(selected)

    pdf_bytes = _build_report_pdf_bytes(selected, ccalc)
    html = _build_report_html(selected, ccalc)
    summary_text = _build_summary_text(selected, ccalc)

    file_base = f"Rental_Income_Report_{_safe_filename(selected.get('Property Name / Address') or '')}"
    pdf_name = f"{file_base}.pdf"

    e1, e2, e3, e4 = st.columns([1, 1, 1, 1])

    with e1:
        st.download_button(
            "⬇️ Report (PDF)",
            data=pdf_bytes,
            file_name=pdf_name,
            mime="application/pdf",
            use_container_width=True,
        )

    with e2:
        st.download_button(
            "⬇️ Table (CSV)",
            data=calc_df.to_csv(index=False).encode("utf-8"),
            file_name="rental_income_table.csv",
            mime="text/csv",
            use_container_width=True,
        )

    subject = f"Rental Income Report - {selected.get('Property Name / Address') or 'Property'}"
    body = (
        "Hi,\n\n"
        "Please find attached the Rental Income Report PDF.\n\n"
        f"Attachment: {pdf_name}\n\n"
        "Regards,\n"
    )
    mailto = "mailto:?" + urllib.parse.urlencode({"subject": subject, "body": body})

    with e3:
        _open_link_button("📧 Email (attach PDF)", mailto, key="rental_email_pdf_btn")

    wa_text = (
        f"Rental Income Report ({_today_long()})\n"
        f"Property: {selected.get('Property Name / Address') or '—'}\n"
        f"Net Monthly: {_fmt_money_dash(ccalc.get('net_m'))}\n"
        f"Net Annual: {_fmt_money_dash(ccalc.get('net_a'))}\n\n"
        f"I can send the PDF as well. Filename: {pdf_name}"
    )
    whatsapp = "https://wa.me/?" + urllib.parse.urlencode({"text": wa_text})

    with e4:
        _open_link_button("💬 WhatsApp (message)", whatsapp, key="rental_whatsapp_msg_btn")

    st.caption("✅ Workflow: Click **Report (PDF)** → then click **Email (attach PDF)** and attach the downloaded PDF.")

    st.text_area(
        "Quick summary (optional)",
        value=summary_text,
        height=140,
        help="This is optional. Your main deliverable is the PDF above.",
    )

    st.markdown("### Report Preview (HTML)")
    st.components.v1.html(html, height=760, scrolling=True)