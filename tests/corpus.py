"""Small multi-domain workspaces, built rather than committed.

Every live bank so far has needed one 2.7GB casino file sitting outside the repo,
which is why the scheduled run wants a self-hosted runner and why nobody else can
reproduce a bank result. These build in memory from a fixed seed, so a bank over
them runs anywhere, and they are small enough that an anchor is an exact figure
rather than a tolerance.

They are shaped, not merely populated. Each carries the hazard its domain runs
into: encounters repeat within a patient, assessments repeat within a student,
a service's latency is skewed so a mean flatters it, and every one of them has a
rate whose denominator is a choice — which is the failure the answer bank reports
more often than any other.

`*.csv` is ignored by this repo on purpose, so nothing here is written to disk.
"""

from __future__ import annotations

import numpy as np

from smart_data_studio.dataset import CsvSource

# One seed per domain, so a change to one leaves the others byte for byte alike.
SEEDS = {
    "healthcare": 11,
    "education": 22,
    "finance": 33,
    "ecommerce": 44,
    "operations": 55,
}


def _csv(name: str, header: str, rows: list[str]) -> CsvSource:
    return CsvSource.from_upload(f"{name}.csv", (header + "\n" + "\n".join(rows) + "\n").encode())


def _day(rng, start: int, span: int) -> str:
    """A date inside 2026, as a plain ISO string DuckDB will read as a DATE."""
    stamp = np.datetime64("2026-01-01") + np.timedelta64(int(start + rng.integers(0, span)), "D")
    return str(stamp)


def healthcare() -> list[CsvSource]:
    """Patients and their encounters.

    The hazard is grain: a patient has many encounters, so a readmission rate per
    patient and per encounter are different numbers, and a significance test over
    encounters treats one patient's six visits as six independent observations.
    """
    rng = np.random.default_rng(SEEDS["healthcare"])
    departments = ["cardiology", "orthopaedics", "general medicine", "respiratory"]
    patients, encounters = [], []
    encounter_id = 1
    for patient_id in range(1, 121):
        band = ["18-39", "40-64", "65-79", "80+"][min(3, int(rng.integers(0, 4)))]
        sex = "F" if patient_id % 2 else "M"
        region = "north" if patient_id % 3 else "south"
        patients.append(f"{patient_id},{band},{sex},{region}")
        # Older bands attend more often, which is what makes per-patient and
        # per-encounter rates diverge rather than merely differ.
        visits = 1 + int(rng.integers(0, 6 if band in ("65-79", "80+") else 3))
        for _ in range(visits):
            department = departments[int(rng.integers(0, len(departments)))]
            admitted = _day(rng, 0, 300)
            stay = 1 + int(rng.gamma(2.0, 2.0))
            readmitted = 1 if rng.random() < (0.28 if band == "80+" else 0.11) else 0
            cost = round(float(600 + stay * rng.normal(950, 220)), 2)
            encounters.append(
                f"{encounter_id},{patient_id},{admitted},{department},{stay},{cost},{readmitted}"
            )
            encounter_id += 1
    return [
        _csv("patients", "patient_id,age_band,sex,region", patients),
        _csv(
            "encounters",
            "encounter_id,patient_id,admitted,department,length_of_stay,cost,readmitted_30d",
            encounters,
        ),
    ]


def education() -> list[CsvSource]:
    """Students and repeated assessments.

    The clustering case in its plainest form: eight scores from one student are
    not eight independent observations, and a pass rate can be counted per sitting
    or per student.
    """
    rng = np.random.default_rng(SEEDS["education"])
    modules = ["algebra", "statistics", "mechanics", "essay"]
    students, assessments = [], []
    for student_id in range(1, 141):
        programme = "foundation" if student_id % 3 == 0 else "honours"
        year = 2024 + int(rng.integers(0, 2))
        students.append(f"{student_id},{year},{programme},{'north' if student_id % 4 else 'south'}")
        # A per-student ability, so the same student's marks move together.
        ability = rng.normal(62 if programme == "honours" else 54, 9)
        for term in (1, 2, 3, 4):
            module = modules[int(rng.integers(0, len(modules)))]
            score = float(np.clip(ability + rng.normal(0, 6), 0, 100))
            assessments.append(
                f"{student_id},{term},{module},{score:.1f},{1 if score >= 50 else 0}"
            )
    return [
        _csv("students", "student_id,cohort_year,programme,region", students),
        _csv("assessments", "student_id,term,module,score,passed", assessments),
    ]


def finance() -> list[CsvSource]:
    """Accounts, their transactions, and which of them defaulted.

    A three-file chain, so a question has to join twice and a measure exists on
    more than one side. The default rate is per account; transactions are per
    account per day, and counting defaults over transactions is the trap.
    """
    rng = np.random.default_rng(SEEDS["finance"])
    products = ["card", "loan", "overdraft"]
    accounts, transactions, defaults = [], [], []
    transaction_id = 1
    for account_id in range(1, 101):
        product = products[int(rng.integers(0, len(products)))]
        segment = "prime" if rng.random() < 0.62 else "subprime"
        accounts.append(f"{account_id},{_day(rng, 0, 120)},{product},{segment}")
        for _ in range(int(rng.integers(3, 30))):
            amount = round(float(rng.gamma(2.0, 90.0)), 2)
            kind = "purchase" if rng.random() < 0.8 else "repayment"
            transactions.append(
                f"{transaction_id},{account_id},{_day(rng, 120, 200)},{amount},{kind}"
            )
            transaction_id += 1
        if rng.random() < (0.24 if segment == "subprime" else 0.06):
            defaults.append(f"{account_id},{_day(rng, 200, 120)}")
    return [
        _csv("accounts", "account_id,opened,product,segment", accounts),
        _csv("transactions", "transaction_id,account_id,day,amount,kind", transactions),
        _csv("defaults", "account_id,defaulted_on", defaults),
    ]


def ecommerce() -> list[CsvSource]:
    """Customers, orders and refunds.

    Order value is skewed, so a mean order value and a median are different
    answers to the same question, and the refund rate can be taken over orders or
    over customers.
    """
    rng = np.random.default_rng(SEEDS["ecommerce"])
    channels = ["search", "email", "social", "direct"]
    customers, orders, refunds = [], [], []
    order_id = 1
    for customer_id in range(1, 151):
        channel = channels[int(rng.integers(0, len(channels)))]
        customers.append(f"{customer_id},{_day(rng, 0, 90)},{channel}")
        for _ in range(int(rng.integers(1, 7))):
            value = round(float(rng.gamma(1.6, 42.0)) + 5, 2)
            orders.append(f"{order_id},{customer_id},{_day(rng, 90, 240)},{value},completed")
            if rng.random() < (0.18 if channel == "social" else 0.07):
                refunds.append(f"{order_id},{round(value, 2)}")
            order_id += 1
    return [
        _csv("customers", "customer_id,joined,channel", customers),
        _csv("orders", "order_id,customer_id,day,value,status", orders),
        _csv("refunds", "order_id,refunded_value", refunds),
    ]


def operations() -> list[CsvSource]:
    """Services and the requests they served.

    Latency is long-tailed, which is the whole point: the mean sits near the body
    and the answer anyone acts on is p95 or p99. The error rate is per request
    against a denominator that must be stated.
    """
    rng = np.random.default_rng(SEEDS["operations"])
    services = ["checkout", "search", "auth", "inventory"]
    rows, catalogue = [], []
    for name in services:
        catalogue.append(f"{name},{'critical' if name in ('checkout', 'auth') else 'standard'}")
    for request_id in range(1, 901):
        service = services[int(rng.integers(0, len(services)))]
        # A body plus a heavy tail, so mean and p99 disagree by an order of magnitude.
        latency = float(
            rng.gamma(2.0, 18.0) + (rng.gamma(2.0, 400.0) if rng.random() < 0.03 else 0)
        )
        failed = 1 if rng.random() < (0.09 if service == "checkout" else 0.02) else 0
        code = 500 if failed else 200
        rows.append(f"{request_id},{service},{_day(rng, 0, 60)},{latency:.1f},{code},{failed}")
    return [
        _csv("services", "service,tier", catalogue),
        _csv("requests", "request_id,service,day,latency_ms,status_code,failed", rows),
    ]


DOMAINS = {
    "healthcare": healthcare,
    "education": education,
    "finance": finance,
    "ecommerce": ecommerce,
    "operations": operations,
}
