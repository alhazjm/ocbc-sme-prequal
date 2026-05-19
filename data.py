"""Data loading: SSIC table + OCBC product catalog + SME profile fixtures."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class SSICEntry:
    code: str
    description: str
    keywords: list[str]


@dataclass
class Product:
    product_name: str
    max_amount_sgd: str
    min_monthly_revenue_sgd: int
    min_years_in_op: float
    indicative_rate_pct: str
    required_documents: str
    source_url: str


@dataclass
class Profile:
    profile_id: str
    industry: str
    ssic_code: str
    monthly_revenue_sgd: int
    years_in_op: float
    employees: int
    loan_purpose: str
    amount_sgd: int
    notes: str


RevenueBand = Literal["under_50k", "50k_200k", "200k_1M", "over_1M"]


def revenue_band(monthly_revenue_sgd: float) -> RevenueBand:
    if monthly_revenue_sgd < 50_000:
        return "under_50k"
    if monthly_revenue_sgd < 200_000:
        return "50k_200k"
    if monthly_revenue_sgd < 1_000_000:
        return "200k_1M"
    return "over_1M"


def band_to_min_monthly(band: RevenueBand) -> int:
    return {"under_50k": 0, "50k_200k": 50_000, "200k_1M": 200_000, "over_1M": 1_000_000}[band]


# Hand-curated SSIC sample covering the common SG SME shapes the agent will see.
# Keywords are matched on word boundaries, so multi-word keywords ("legal services")
# must appear verbatim. Avoid single-letter or 2-char keywords — they cause false matches.
SSIC_TABLE: list[SSICEntry] = [
    SSICEntry("47190", "Retail sale in non-specialised stores", ["retail", "shop", "store", "minimart"]),
    SSICEntry("47211", "Retail sale of food in specialised stores (supermarket)", ["supermarket", "grocery"]),
    SSICEntry("47410", "Retail sale of computer hardware", ["computer hardware", "electronics retail"]),
    SSICEntry("56111", "Restaurants", ["restaurant", "f&b", "food and beverage", "dining", "eatery"]),
    SSICEntry("56121", "Cafés and coffee houses", ["café", "cafe", "coffee shop", "bakery"]),
    SSICEntry("56210", "Event catering", ["catering", "event catering"]),
    SSICEntry("62019", "Computer programming activities (software dev)", ["software", "saas", "software development", "engineering platform", "app development"]),
    SSICEntry("62022", "IT consultancy", ["it consulting", "it consultancy", "tech consulting"]),
    SSICEntry("73100", "Advertising / marketing services", ["marketing", "advertising", "creative agency", "ad agency", "public relations", "branding"]),
    SSICEntry("70209", "Management consultancy", ["management consulting", "strategy consulting", "business consulting", "management consultancy"]),
    SSICEntry("49231", "Freight transport by road", ["logistics", "trucking", "freight", "haulage"]),
    SSICEntry("49100", "Passenger land transport", ["passenger transport", "taxi", "bus operator"]),
    SSICEntry("25910", "Treatment and coating of metals", ["manufacturing", "metal fabrication", "machining", "fabrication"]),
    SSICEntry("41001", "General building contractors (residential)", ["construction", "contractor", "building contractor", "renovation"]),
    SSICEntry("96911", "Hairdressing and beauty salons", ["salon", "hairdressing", "beauty salon", "spa"]),
    SSICEntry("86201", "Medical and dental practice", ["clinic", "medical practice", "dental", "healthcare"]),
    SSICEntry("85501", "Tuition services", ["tuition", "tutoring", "enrichment centre"]),
    SSICEntry("95210", "Repair of consumer electronics", ["electronics repair", "phone repair"]),
    SSICEntry("69100", "Legal services / legal activities", ["legal services", "legal activities", "law firm", "legal consultancy", "legal managed services", "legal process outsourcing", "lpo"]),
    SSICEntry("69202", "Bookkeeping and other accounting activities", ["bookkeeping", "accounting services", "tax consultancy", "audit services"]),
    SSICEntry("82300", "Corporate secretarial and convention support", ["corporate secretarial", "company secretary", "compliance services", "corporate governance"]),
    SSICEntry("82990", "Other business support services / BPO", ["bpo", "business process outsourcing", "outsourcing", "business support services", "back office"]),
]


def _to_int(s: str | int | float, default: int = 0) -> int:
    try:
        return int(float(str(s)))
    except (ValueError, TypeError):
        return default


def _to_float(s: str | int | float, default: float = 0.0) -> float:
    try:
        return float(str(s))
    except (ValueError, TypeError):
        return default


def load_products() -> list[Product]:
    out: list[Product] = []
    with open(DATA_DIR / "ocbc_products.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                Product(
                    product_name=row["product_name"].strip(),
                    max_amount_sgd=row["max_amount_sgd"].strip(),
                    min_monthly_revenue_sgd=_to_int(row["min_monthly_revenue_sgd"]),
                    min_years_in_op=_to_float(row["min_years_in_op"]),
                    indicative_rate_pct=row["indicative_rate_pct"].strip(),
                    required_documents=row["required_documents"].strip(),
                    source_url=row["source_url"].strip(),
                )
            )
    return out


def load_profiles() -> list[Profile]:
    out: list[Profile] = []
    with open(DATA_DIR / "sme_profiles.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                Profile(
                    profile_id=row["profile_id"].strip(),
                    industry=row["industry"].strip(),
                    ssic_code=row["ssic_code"].strip(),
                    monthly_revenue_sgd=_to_int(row["monthly_revenue_sgd"]),
                    years_in_op=_to_float(row["years_in_op"]),
                    employees=_to_int(row["employees"]),
                    loan_purpose=row["loan_purpose"].strip(),
                    amount_sgd=_to_int(row["amount_sgd"]),
                    notes=row["notes"].strip(),
                )
            )
    return out


PRODUCTS: list[Product] = load_products()
PROFILES: list[Profile] = load_profiles()
