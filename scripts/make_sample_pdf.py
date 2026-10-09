"""Generate the sample PDF used by the demo (a FICTIONAL bank policy).

Run:  python scripts/make_sample_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_docs" / "northwind_home_loan_policy.pdf"

SECTIONS = [
    ("1. Purpose and Scope", [
        "This policy sets out how Northwind Bank (a fictional bank created for this demo) assesses, "
        "approves and services home loans for resident individuals. It applies to purchase of a new or "
        "resale residential property, construction on an owned plot, and balance transfer from another lender.",
        "The policy is reviewed every twelve months by the Credit Risk Committee. Exceptions require written "
        "approval from the Chief Credit Officer.",
    ]),
    ("2. Eligibility", [
        "Applicants must be between 21 and 65 years of age at the time of application, and the loan must be "
        "fully repaid before the borrower turns 70.",
        "Salaried applicants need at least two years of total work experience, with a minimum of six months "
        "in the current job. Self-employed applicants need three years of continuous business history, "
        "supported by audited financial statements.",
        "The minimum credit bureau score is 700. Applications with a score between 650 and 699 may be "
        "considered with a higher interest rate and a co-applicant. Scores below 650 are declined.",
    ]),
    ("3. Loan Amount and Loan-to-Value (LTV)", [
        "The maximum loan-to-value ratio depends on the loan amount: up to 90% for loans up to Rs 30 lakh, "
        "up to 80% for loans between Rs 30 lakh and Rs 75 lakh, and up to 75% for loans above Rs 75 lakh.",
        "Stamp duty and registration charges are not included in the property value when computing LTV.",
        "The total of all monthly loan instalments (EMIs), including the new home loan, must not exceed 50% "
        "of the applicant's net monthly income. This is called the Fixed Obligation to Income Ratio (FOIR).",
    ]),
    ("4. Interest Rates", [
        "Home loans are offered on a floating rate linked to the bank's External Benchmark Lending Rate (EBLR). "
        "The rate is reset every three months when the benchmark changes.",
        "The spread over EBLR is decided by the applicant's credit score: 0.50% for scores of 800 and above, "
        "0.75% for scores between 750 and 799, and 1.10% for scores between 700 and 749.",
        "Women borrowers, as sole or first applicant, receive a concession of 0.05% on the spread.",
    ]),
    ("5. Required Documents", [
        "All applicants must submit: a completed application form with photograph; proof of identity and "
        "address; bank statements for the last six months; and the property documents, including the sale "
        "agreement and approved building plan.",
        "Salaried applicants must also submit salary slips for the last three months and Form 16 for the last "
        "two years. Self-employed applicants must submit income tax returns for the last three years and "
        "audited profit and loss statements.",
    ]),
    ("6. Prepayment and Foreclosure", [
        "Borrowers on a floating rate may prepay part or all of the loan at any time with no prepayment "
        "penalty. Part-prepayments must be at least one EMI in value.",
        "After a part-prepayment the borrower may choose to reduce either the EMI or the remaining tenure. "
        "If no choice is made, the bank reduces the tenure.",
        "On full foreclosure, the bank returns the original property documents within 30 days of final payment.",
    ]),
    ("7. Missed Payments and Collections", [
        "If an EMI is not paid by the due date, a late payment charge of 2% per month is levied on the overdue "
        "amount. The borrower is sent an SMS and email reminder on the first day after the due date.",
        "An account with any amount overdue for more than 90 days is classified as a Non-Performing Asset "
        "(NPA), in line with regulatory guidelines, and is moved to the collections team.",
        "Under the bank's expected credit loss model, an account more than 30 days past due is treated as a "
        "significant increase in credit risk (Stage 2), and more than 90 days past due as credit-impaired (Stage 3).",
    ]),
    ("8. Grievance Redressal", [
        "Customers may raise complaints at any branch, by email, or through the mobile app. The bank "
        "acknowledges every complaint within 2 working days and aims to resolve it within 15 working days.",
        "If the customer is not satisfied with the resolution, or receives no reply within 30 days, they may "
        "escalate the complaint to the banking ombudsman.",
    ]),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Northwind Bank - Home Loan Policy (Fictional Sample)", styles["Title"]),
        Paragraph("Version 3.2 · For demonstration only. Not a real bank or real policy.", styles["Italic"]),
        Spacer(1, 16),
    ]
    for i, (heading, paragraphs) in enumerate(SECTIONS):
        if i in (3, 6):  # spread sections over several pages so page citations are meaningful
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading2"]))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 6))

    OUT.parent.mkdir(exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, title="Northwind Bank Home Loan Policy (Fictional)").build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
