from __future__ import annotations

from fpdf import FPDF


class HandbookPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.page_title, ln=True)
        self.ln(2)


def add_section_title(pdf: HandbookPDF, title: str):
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, title)
    pdf.ln(1)


def add_paragraph(pdf: HandbookPDF, text: str):
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, text)
    pdf.ln(1)


def add_table(pdf: HandbookPDF, headers: list[str], rows: list[list[str]]):
    pdf.set_font("Helvetica", "B", 10)
    col_widths = [40, 70, 40, 40]
    for idx, header in enumerate(headers):
        pdf.cell(col_widths[idx], 8, header, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for row in rows:
        for idx, cell in enumerate(row):
            pdf.cell(col_widths[idx], 8, cell, border=1)
        pdf.ln()
    pdf.ln(2)


def build_pdf(output_path: str) -> None:
    pdf = HandbookPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # PAGE 1
    pdf.page_title = "TechCorp Global - Employee Handbook 2025"
    pdf.add_page()
    add_section_title(pdf, "1. Mission Statement")
    add_paragraph(
        pdf,
        "TechCorp aims to revolutionize the enterprise AI space. We value "
        "transparency, integrity, and innovation above all. Our goal is to "
        "empower businesses with decision intelligence.",
    )

    add_section_title(pdf, "2. Work Arrangements & Remote Policy")
    add_paragraph(pdf, "Standard Hours: 09:00 - 18:00 EST.")
    add_paragraph(
        pdf,
        "Remote Work: Employees are permitted to work remotely up to 2 days a week "
        "(Tuesday and Thursday preferred).",
    )
    add_paragraph(
        pdf,
        'Full Remote Exception: Engineering teams may work fully remote during '
        'designated "Sprint Crunch" periods, subject to direct CTO approval.',
    )
    add_paragraph(
        pdf,
        'Office Presence: All employees must be in the office on Mondays for the '
        '"All-Hands" meeting.',
    )

    add_section_title(pdf, "3. IT & Security Policy")
    add_paragraph(
        pdf,
        'Password Rotation: All employees must reset their passwords every 90 days '
        'via the "SecureID Portal".',
    )
    add_paragraph(
        pdf,
        "Device Policy: Personal laptops are strictly prohibited on the corporate "
        "network (TechCorp-Secure).",
    )
    add_paragraph(
        pdf,
        'Data Classification: Do not share "Level 3" documents with interns or '
        "contractors.",
    )

    # PAGE 2
    pdf.page_title = "TechCorp Global - Benefits & Compensation"
    pdf.add_page()
    add_section_title(pdf, "4. Benefits Structure")
    add_section_title(
        pdf,
        "4.1 Health Insurance Tiers Eligibility Note: Coverage begins strictly "
        "after the completion of the 90-day probationary period.",
    )
    add_table(
        pdf,
        headers=[
            "Tier Name",
            "Coverage Details",
            "Annual Deductible",
            "Employee Monthly Cost",
        ],
        rows=[
            ["Bronze", "Basic Health & Emergency", "$5,000", "$0 (Free)"],
            ["Silver", "Health + Dental + Vision", "$2,000", "$150"],
            ["Gold", "Full Comprehensive + Family", "$500", "$400"],
        ],
    )

    add_section_title(pdf, "4.2 Leave Policy (PTO)")
    add_paragraph(pdf, "Vacation: 20 days per year (accrued monthly).")
    add_paragraph(pdf, "Sick Leave: 10 days per year.")
    add_paragraph(
        pdf,
        "Sabbatical: Employees are eligible for a 4-week paid sabbatical after "
        "5 years of continuous service.",
    )
    add_paragraph(
        pdf,
        "Unpaid Leave: Requires written approval from the VP of HR.",
    )

    # PAGE 3
    pdf.page_title = "TechCorp Global - Executive Compensation"
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(200, 0, 0)
    pdf.multi_cell(
        0, 10, "CONFIDENTIAL - INTERNAL USE ONLY - LEVEL 3 ACCESS"
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    add_section_title(pdf, "5. Executive Compensation & Strategy (Admin/Exec Only)")
    add_section_title(
        pdf,
        "5.1 Salary Bands (2025 Adjustment) The following bands are strictly for "
        "HR planning and Executive review.",
    )
    add_table(
        pdf,
        headers=["Level", "Role Title", "Base Salary Range", "Stock Options (RSU)"],
        rows=[
            ["L1", "Junior Engineer", "$70,000 - $90,000", "500 units"],
            ["L2", "Senior Engineer", "$140,000 - $180,000", "2,500 units"],
            ["L3", "Staff Engineer", "$220,000 - $280,000", "5,000 units"],
            ["M1", "Engineering Manager", "$190,000 - $230,000", "4,000 units"],
            ["E1", "VP of Engineering", "$350,000+", "15,000 units"],
        ],
    )

    add_section_title(pdf, "5.2 Executive Bonus Structure")
    add_paragraph(pdf, "CEO Base Salary: $600,000")
    add_paragraph(
        pdf,
        "Performance Bonus Formula: The CEO is eligible for a year-end cash bonus "
        "calculated as 3.5% of the company's Annual Net Profit.",
    )
    add_paragraph(pdf, "2025 Financial Projections:")
    add_paragraph(pdf, "Projected Revenue: $120,000,000")
    add_paragraph(pdf, "Projected Net Profit: $40,000,000")
    add_paragraph(pdf, "Payment Date: Bonuses are paid out on Q1 of the following fiscal year.")

    add_section_title(pdf, "5.3 Strategic Acquisitions (Top Secret)")
    add_paragraph(
        pdf,
        'Target: TechCorp is currently in silent negotiations to acquire "DataAI Inc."',
    )
    add_paragraph(pdf, "Offer Price: $15,000,000 (Cash + Stock).")
    add_paragraph(pdf, "Timeline: Expected closing date is Q3 2025.")
    add_paragraph(
        pdf,
        "Action Item: Due diligence is ongoing. Do not discuss with DataAI employees.",
    )

    pdf.output(output_path)


if __name__ == "__main__":
    build_pdf("TechCorp_Global_Handbook_2025.pdf")
