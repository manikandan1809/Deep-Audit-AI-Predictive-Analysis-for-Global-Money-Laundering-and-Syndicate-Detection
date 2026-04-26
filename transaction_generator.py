"""
==============================================================================
  DEEP AUDIT AI  —  LIVE TRANSACTION GENERATOR
  Simulates a real bank's Core Banking System (CBS)
  
  Run this in a SEPARATE terminal:
      python transaction_generator.py
  
  Then run the main app in another terminal:
      streamlit run app.py
  
  This generator:
  - Creates realistic Indian banking transactions every 2-5 seconds
  - Injects real fraud patterns (smurfing, layering, integration, hawala)
  - Writes to SQLite database that the Streamlit app reads live
  - Simulates 5 banks, 200+ accounts, 9 countries
  - Randomly triggers syndicate bursts (the "boss" scenario)
==============================================================================
"""

import sqlite3
import random
import time
import datetime
import uuid
import math
import signal
import sys

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH           = "live_transactions.db"
TRANSACTIONS_PER_SECOND = 0.4    # ~1 transaction every 2.5 seconds (realistic for demo)
FRAUD_INJECTION_RATE    = 0.08   # 8% of transactions are fraud
SYNDICATE_BURST_CHANCE  = 0.03   # 3% chance of triggering a full syndicate burst

# ── Realistic Indian Bank Data ─────────────────────────────────────────────────
BANKS = [
    "SBI", "HDFC", "ICICI", "AXIS", "PNB",
    "BOB", "CANARA", "KOTAK", "YES", "INDUSIND"
]

TRANSACTION_TYPES = ["NEFT", "RTGS", "UPI", "IMPS", "WIRE", "SWIFT", "CASH_DEPOSIT"]

DOMESTIC_LOCATIONS = [
    "Mumbai", "Delhi", "Chennai", "Bangalore", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Surat", "Nagpur", "Indore", "Bhopal", "Patna"
]

INTERNATIONAL_LOCATIONS = [
    "Dubai, UAE", "Singapore", "London, UK", "Zurich, Switzerland",
    "Cayman Islands", "British Virgin Islands", "Mauritius",
    "Hong Kong", "Panama City, Panama", "Nicosia, Cyprus"
]

NORMAL_DESCRIPTIONS = [
    "Salary credit", "EMI payment", "Insurance premium",
    "Mutual fund investment", "Utility bill payment",
    "Online shopping", "School fee", "Medical expense",
    "Rent payment", "Grocery purchase", "Fuel purchase",
    "Tax payment", "LIC premium", "PPF contribution"
]

FRAUD_DESCRIPTIONS = {
    "SMURFING":    ["Split cash deposit", "Structured deposit below limit",
                    "Multiple small transfers", "Threshold avoidance transfer"],
    "LAYERING":    ["Transit account routing", "Shell company transfer",
                    "Nominee account transfer", "Syndicate fund routing",
                    "Intermediary layering"],
    "INTEGRATION": ["Offshore wire transfer", "Final integration transfer",
                    "Swift transfer to boss account", "Offshore consolidation"],
    "HAWALA":      ["Hawala network transfer", "Informal value transfer",
                    "Hundi transaction", "Undocumented cross-border"],
    "TERROR":      ["Sanctioned entity payment", "Frozen asset transfer",
                    "Terror financing route"]
}

# ── Synthetic Account Pools ────────────────────────────────────────────────────
def _make_account(prefix, n):
    return [f"{prefix}_{str(i).zfill(4)}" for i in range(1, n + 1)]

NORMAL_ACCOUNTS  = _make_account("ACC_NORMAL",  150)
MULE_ACCOUNTS    = _make_account("ACC_MULE",     30)
SHELL_ACCOUNTS   = _make_account("SHELL_ACC",    20)
BOSS_ACCOUNTS    = ["LEODASS_BB_2024", "REDDRAGON_BB_2025", "MANI_SCAMMER_2026", "MANOJ_HACKER_2026", "SIMON_NTK_2026"]

# ── KYC Profile data pools ─────────────────────────────────────────────────────
_FIRST_NAMES = ["Rajesh","Priya","Suresh","Anitha","Vijay","Deepa","Karthik",
                "Meena","Arun","Kavitha","Ramesh","Sunita","Manoj","Lakshmi",
                "Sanjay","Pooja","Dinesh","Nirmala","Prakash","Geetha",
                "Mohammed","Fatima","Ravi","Saranya","Balu","Revathi"]
_LAST_NAMES  = ["Kumar","Sharma","Patel","Reddy","Nair","Pillai","Murugan",
                "Singh","Rao","Iyer","Krishnan","Gupta","Bose","Menon",
                "Chandra","Verma","Shah","Joshi","Das","Roy","Khan","Siddiqui"]
_CITIES_KYC  = ["Mumbai","Delhi","Chennai","Bangalore","Hyderabad","Pune",
                 "Kolkata","Ahmedabad","Jaipur","Lucknow","Coimbatore","Madurai"]
_RISK_LEVELS = ["LOW","LOW","LOW","MEDIUM","MEDIUM","HIGH","HIGH","CRITICAL"]

def _random_pan():
    import random, string
    return (random.choice(string.ascii_uppercase) * 5 +
            str(random.randint(1000, 9999)) +
            random.choice(string.ascii_uppercase)).replace(' ','')

def _random_phone():
    import random
    return "9" + str(random.randint(100_000_000, 999_999_999))

def _random_pan_proper():
    """PAN format: AAAAA9999A"""
    import random, string
    letters = string.ascii_uppercase
    return (
        ''.join(random.choices(letters, k=5)) +
        str(random.randint(1000, 9999)) +
        random.choice(letters)
    )

def _random_aadhar():
    import random
    return f"{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

def _build_kyc_profile(account_id, role="NORMAL"):
    """Build a synthetic KYC profile for an account."""
    import random
    rng = random.Random(hash(account_id) % (2**31))  # deterministic per account
    first  = rng.choice(_FIRST_NAMES)
    last   = rng.choice(_LAST_NAMES)
    name   = f"{first} {last}"
    city   = rng.choice(_CITIES_KYC)
    state  = rng.choice(["Tamil Nadu","Maharashtra","Karnataka","Delhi","Gujarat","Rajasthan"])
    pin    = str(rng.randint(400001, 600099))
    phone  = "9" + str(rng.randint(100_000_000, 999_999_999))
    email  = f"{first.lower()}.{last.lower()}{rng.randint(1,999)}@{'gmail' if rng.random()<0.6 else 'yahoo'}.com"
    pan    = (''.join([chr(65+rng.randint(0,25)) for _ in range(5)]) +
              str(rng.randint(1000,9999)) + chr(65+rng.randint(0,25)))
    aadhar = f"{rng.randint(1000,9999)} {rng.randint(1000,9999)} {rng.randint(1000,9999)}"
    dob    = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1960,1998)}"
    addr   = (f"{rng.randint(1,999)}, {rng.choice(['MG Road','Anna Salai','Gandhi Nagar','Nehru Street','Rajiv Colony'])},"
              f" {city}, {state} - {pin}")
    bank   = rng.choice(BANKS)

    # Risk level based on role
    if role == "BOSS":
        risk = "CRITICAL"
    elif role == "SHELL":
        risk = "HIGH"
    elif role == "MULE":
        risk = rng.choice(["MEDIUM","HIGH"])
    else:
        risk = rng.choice(["LOW","LOW","LOW","MEDIUM"])

    occupation = rng.choice([
        "Software Engineer","Business Owner","Teacher","Doctor","Trader",
        "Real Estate Agent","Import/Export","Consultant","Retired",
        "Student","Shopkeeper","Contractor"
    ])
    annual_income = rng.choice([
        "₹2.5L - ₹5L","₹5L - ₹10L","₹10L - ₹25L",
        "₹25L - ₹50L","₹50L+","Not Disclosed"
    ])
    return {
        "Account_ID"    : account_id,
        "Holder_Name"   : name,
        "Phone"         : phone,
        "Email"         : email,
        "PAN"           : pan,
        "Aadhar"        : aadhar,
        "DOB"           : dob,
        "Address"       : addr,
        "City"          : city,
        "State"         : state,
        "Bank_Name"     : bank,
        "Occupation"    : occupation,
        "Annual_Income" : annual_income,
        "Risk_Level"    : risk,
        "Account_Role"  : role,
        "Registered_On" : f"2022-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "KYC_Status"    : rng.choice(["COMPLETE","COMPLETE","COMPLETE","PENDING","FLAGGED"]),
    }

# Current active boss (rotates every ~500 transactions for realism)
_active_boss = BOSS_ACCOUNTS[0]
_txn_count   = 0

# ── Database Setup ─────────────────────────────────────────────────────────────
def init_database():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    for pragma in [
        "PRAGMA busy_timeout = 30000",
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = NORMAL",
    ]:
        try:
            conn.execute(pragma)
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_transactions (
            Transaction_ID      TEXT PRIMARY KEY,
            Timestamp           TEXT NOT NULL,
            Source_Acc_No       TEXT NOT NULL,
            Dest_Acc_No         TEXT NOT NULL,
            Amount_INR          REAL NOT NULL,
            Bank_Name           TEXT NOT NULL,
            Transaction_Type    TEXT NOT NULL,
            Is_International    INTEGER DEFAULT 0,
            Location            TEXT DEFAULT 'Unknown',
            Fraud_Label         TEXT DEFAULT 'NORMAL',
            Txn_Description     TEXT DEFAULT '',
            IP_Address          TEXT DEFAULT '0.0.0.0',
            Exchange_Rate       REAL DEFAULT 1.0,
            inserted_at         TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generator_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            total_sent  INTEGER,
            fraud_sent  INTEGER,
            last_boss   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kyc_profiles (
            Account_ID      TEXT PRIMARY KEY,
            Holder_Name     TEXT,
            Phone           TEXT,
            Email           TEXT,
            PAN             TEXT,
            Aadhar          TEXT,
            DOB             TEXT,
            Address         TEXT,
            City            TEXT,
            State           TEXT,
            Bank_Name       TEXT,
            Occupation      TEXT,
            Annual_Income   TEXT,
            Risk_Level      TEXT,
            Account_Role    TEXT,
            Registered_On   TEXT,
            KYC_Status      TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_transactions_inserted_at
        ON live_transactions (inserted_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_transactions_source
        ON live_transactions (Source_Acc_No)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_transactions_dest
        ON live_transactions (Dest_Acc_No)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_transactions_fraud
        ON live_transactions (Fraud_Label)
    """)
    conn.commit()
    _seed_kyc_profiles(conn)
    return conn


def _seed_kyc_profiles(conn):
    """Pre-populate KYC profiles for all known accounts."""
    existing = {row[0] for row in conn.execute("SELECT Account_ID FROM kyc_profiles").fetchall()}
    rows = []
    for acc in NORMAL_ACCOUNTS:
        if acc not in existing:
            rows.append(_build_kyc_profile(acc, "NORMAL"))
    for acc in MULE_ACCOUNTS:
        if acc not in existing:
            rows.append(_build_kyc_profile(acc, "MULE"))
    for acc in SHELL_ACCOUNTS:
        if acc not in existing:
            rows.append(_build_kyc_profile(acc, "SHELL"))
    for acc in BOSS_ACCOUNTS:
        if acc not in existing:
            rows.append(_build_kyc_profile(acc, "BOSS"))
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO kyc_profiles VALUES "
            "(:Account_ID,:Holder_Name,:Phone,:Email,:PAN,:Aadhar,:DOB,"
            ":Address,:City,:State,:Bank_Name,:Occupation,:Annual_Income,"
            ":Risk_Level,:Account_Role,:Registered_On,:KYC_Status)",
            rows)
        conn.commit()
        print(f"  [KYC] Seeded {len(rows)} account profiles")


def generate_normal_transaction():
    """Generate a realistic normal banking transaction."""
    src  = random.choice(NORMAL_ACCOUNTS)
    dest = random.choice([a for a in NORMAL_ACCOUNTS if a != src])
    bank = random.choice(BANKS)

    # Normal amounts: salary/rent/EMI range
    amount = random.choice([
        random.uniform(500, 9_999),       # small daily
        random.uniform(10_000, 49_999),   # medium
        random.uniform(50_000, 2_00_000), # large legit
    ])

    is_intl   = 0
    location  = random.choice(DOMESTIC_LOCATIONS)
    txn_type  = random.choice(["NEFT", "RTGS", "UPI", "IMPS", "CASH_DEPOSIT"])
    desc      = random.choice(NORMAL_DESCRIPTIONS)
    ip        = f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
    timestamp = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 30))

    return {
        "Transaction_ID"  : f"TXN_{uuid.uuid4().hex[:12].upper()}",
        "Timestamp"       : timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Source_Acc_No"   : src,
        "Dest_Acc_No"     : dest,
        "Amount_INR"      : round(amount, 2),
        "Bank_Name"       : bank,
        "Transaction_Type": txn_type,
        "Is_International": is_intl,
        "Location"        : location,
        "Fraud_Label"     : "NORMAL",
        "Txn_Description" : desc,
        "IP_Address"      : ip,
        "Exchange_Rate"   : 1.0,
    }


def generate_smurfing_burst(boss):
    """
    Smurfing: 5–15 mule accounts each send small amounts
    (just below ₹50,000 threshold) to shell accounts.
    """
    transactions = []
    n_mules  = random.randint(5, 15)
    mules    = random.sample(MULE_ACCOUNTS, min(n_mules, len(MULE_ACCOUNTS)))
    shells   = random.sample(SHELL_ACCOUNTS, min(3, len(SHELL_ACCOUNTS)))
    bank     = random.choice(BANKS)
    now      = datetime.datetime.now()

    for i, mule in enumerate(mules):
        shell  = random.choice(shells)
        amount = random.uniform(45_000, 49_500)  # Just below ₹50,000 reporting threshold
        ts     = now + datetime.timedelta(seconds=i * random.randint(10, 45))
        transactions.append({
            "Transaction_ID"  : f"TXN_{uuid.uuid4().hex[:12].upper()}",
            "Timestamp"       : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Source_Acc_No"   : mule,
            "Dest_Acc_No"     : shell,
            "Amount_INR"      : round(amount, 2),
            "Bank_Name"       : bank,
            "Transaction_Type": random.choice(["NEFT", "CASH_DEPOSIT"]),
            "Is_International": 0,
            "Location"        : random.choice(DOMESTIC_LOCATIONS),
            "Fraud_Label"     : "SMURFING",
            "Txn_Description" : random.choice(FRAUD_DESCRIPTIONS["SMURFING"]),
            "IP_Address"      : f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
            "Exchange_Rate"   : 1.0,
        })
    return transactions


def generate_layering_chain(boss):
    """
    Layering: shell → shell → shell chain.
    Money hops through 3–6 shell accounts to obscure trail.
    """
    transactions = []
    chain_length = random.randint(3, 6)
    shells       = random.sample(SHELL_ACCOUNTS, min(chain_length, len(SHELL_ACCOUNTS)))
    bank         = random.choice(BANKS)
    now          = datetime.datetime.now()
    amount       = random.uniform(2_00_000, 15_00_000)

    for i in range(len(shells) - 1):
        src  = shells[i]
        dest = shells[i + 1]
        ts   = now + datetime.timedelta(minutes=i * random.randint(5, 20))
        transactions.append({
            "Transaction_ID"  : f"TXN_{uuid.uuid4().hex[:12].upper()}",
            "Timestamp"       : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Source_Acc_No"   : src,
            "Dest_Acc_No"     : dest,
            "Amount_INR"      : round(amount * random.uniform(0.88, 0.98), 2),
            "Bank_Name"       : bank,
            "Transaction_Type": "WIRE",
            "Is_International": 0,
            "Location"        : random.choice(DOMESTIC_LOCATIONS),
            "Fraud_Label"     : "LAYERING",
            "Txn_Description" : random.choice(FRAUD_DESCRIPTIONS["LAYERING"]),
            "IP_Address"      : f"172.16.{random.randint(1,254)}.{random.randint(1,254)}",
            "Exchange_Rate"   : 1.0,
        })
    return transactions


def generate_integration_transfer(boss):
    """
    Integration: Final large transfer to offshore boss account.
    This is what the PageRank model identifies as the convergence node.
    """
    shell  = random.choice(SHELL_ACCOUNTS)
    amount = random.uniform(50_00_000, 3_10_00_000)  # ₹50L to ₹3.1Cr
    intl_loc = random.choice(INTERNATIONAL_LOCATIONS)

    return [{
        "Transaction_ID"  : f"TXN_{uuid.uuid4().hex[:12].upper()}",
        "Timestamp"       : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source_Acc_No"   : shell,
        "Dest_Acc_No"     : boss,
        "Amount_INR"      : round(amount, 2),
        "Bank_Name"       : random.choice(BANKS),
        "Transaction_Type": random.choice(["SWIFT", "WIRE"]),
        "Is_International": 1,
        "Location"        : intl_loc,
        "Fraud_Label"     : "INTEGRATION",
        "Txn_Description" : random.choice(FRAUD_DESCRIPTIONS["INTEGRATION"]),
        "IP_Address"      : f"203.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        "Exchange_Rate"   : round(random.uniform(82, 86), 4),
    }]


def generate_hawala_transaction():
    """Hawala: informal cross-border value transfer."""
    src   = random.choice(MULE_ACCOUNTS)
    dest  = random.choice(SHELL_ACCOUNTS + BOSS_ACCOUNTS[:1])
    amount = random.uniform(1_00_000, 20_00_000)

    return [{
        "Transaction_ID"  : f"TXN_{uuid.uuid4().hex[:12].upper()}",
        "Timestamp"       : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source_Acc_No"   : src,
        "Dest_Acc_No"     : dest,
        "Amount_INR"      : round(amount, 2),
        "Bank_Name"       : random.choice(BANKS),
        "Transaction_Type": "HAWALA",
        "Is_International": 1,
        "Location"        : random.choice(INTERNATIONAL_LOCATIONS),
        "Fraud_Label"     : "HAWALA",
        "Txn_Description" : random.choice(FRAUD_DESCRIPTIONS["HAWALA"]),
        "IP_Address"      : f"198.51.{random.randint(1,254)}.{random.randint(1,254)}",
        "Exchange_Rate"   : round(random.uniform(82, 86), 4),
    }]


def generate_full_syndicate_burst(boss):
    """
    Full syndicate cycle: smurfing → layering → integration
    All three layers in one coordinated burst.
    This is the scenario your project is designed to catch.
    """
    all_txns = []
    all_txns.extend(generate_smurfing_burst(boss))
    all_txns.extend(generate_layering_chain(boss))
    all_txns.extend(generate_integration_transfer(boss))
    return all_txns


def insert_transactions(conn, transactions):
    """Insert transactions into SQLite live database."""
    now = datetime.datetime.now().isoformat(timespec='seconds')
    rows = [(
        t["Transaction_ID"],
        t["Timestamp"],
        t["Source_Acc_No"],
        t["Dest_Acc_No"],
        float(t["Amount_INR"]),
        t["Bank_Name"],
        t["Transaction_Type"],
        int(t["Is_International"]),
        t["Location"],
        t["Fraud_Label"],
        t["Txn_Description"],
        t["IP_Address"],
        float(t["Exchange_Rate"]),
        now,
    ) for t in transactions]

    for attempt in range(3):
        try:
            conn.executemany("""
                INSERT OR IGNORE INTO live_transactions
                (Transaction_ID, Timestamp, Source_Acc_No, Dest_Acc_No, Amount_INR,
                 Bank_Name, Transaction_Type, Is_International, Location, Fraud_Label,
                 Txn_Description, IP_Address, Exchange_Rate, inserted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if 'locked' not in str(exc).lower() or attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))


def print_status(txn, is_fraud=False, burst_type=None):
    """Print real-time status to terminal."""
    ts     = txn["Timestamp"]
    amt    = f"Rs.{txn['Amount_INR']:>12,.0f}"
    src    = txn["Source_Acc_No"][:18].ljust(18)
    dest   = txn["Dest_Acc_No"][:18].ljust(18)
    label  = txn["Fraud_Label"]

    if label == "NORMAL":
        status = "  ✓ NORMAL  "
        color  = "\033[32m"    # green
    elif label == "SMURFING":
        status = "⚡ SMURFING "
        color  = "\033[33m"    # yellow
    elif label == "LAYERING":
        status = "🔗 LAYERING "
        color  = "\033[35m"    # magenta
    elif label == "INTEGRATION":
        status = "🎯 INTEGR.  "
        color  = "\033[31m"    # red
    elif label == "HAWALA":
        status = "💰 HAWALA   "
        color  = "\033[36m"    # cyan
    else:
        status = "⚠ UNKNOWN  "
        color  = "\033[37m"

    reset  = "\033[0m"
    txn_id = txn["Transaction_ID"]   # full ID

    if burst_type:
        print(f"\n{'='*80}")
        print(f"  🚨  SYNDICATE BURST TRIGGERED: {burst_type.upper()}")
        print(f"{'='*80}")

    print(f"{color}[{ts}]  {status}  TXN:{txn_id}  |  {src} → {dest}  |  {amt}  |  [{txn['Bank_Name']}]{reset}")


# ── Main Generator Loop ────────────────────────────────────────────────────────
def run():
    global _active_boss, _txn_count

    print("\n" + "="*80)
    print("  DEEP AUDIT AI — LIVE TRANSACTION GENERATOR")
    print("  Simulating bank Core Banking System (CBS)")
    print(f"  Database: {DB_PATH}")
    print(f"  Rate: ~{TRANSACTIONS_PER_SECOND} transactions/second")
    print(f"  Fraud injection rate: {FRAUD_INJECTION_RATE*100:.0f}%")
    print("="*80)
    print("  Press Ctrl+C to stop\n")
    print("  COLOR CODE:")
    print("  \033[32m✓ GREEN  = Normal transaction\033[0m")
    print("  \033[33m⚡ YELLOW = Smurfing\033[0m")
    print("  \033[35m🔗 MAGENTA= Layering\033[0m")
    print("  \033[31m🎯 RED   = Integration (boss transfer)\033[0m")
    print("  \033[36m💰 CYAN  = Hawala\033[0m")
    print("\n" + "-"*80 + "\n")

    conn          = init_database()
    total_sent    = 0
    fraud_sent    = 0
    delay         = 1.0 / TRANSACTIONS_PER_SECOND

    def shutdown(sig, frame):
        print(f"\n\n{'='*80}")
        print(f"  Generator stopped.")
        print(f"  Total transactions sent : {total_sent:,}")
        print(f"  Fraud transactions sent : {fraud_sent:,}")
        print(f"  Fraud rate              : {fraud_sent/max(1,total_sent)*100:.1f}%")
        print(f"  Database saved to       : {DB_PATH}")
        print(f"{'='*80}\n")
        conn.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, shutdown)

    while True:
        _txn_count += 1

        # Rotate boss account every 300 transactions (realism)
        if _txn_count % 300 == 0:
            _active_boss = random.choice(BOSS_ACCOUNTS)
            print(f"\n  ⟳ Active boss rotated → {_active_boss}\n")

        transactions_to_insert = []
        burst_type             = None

        # Decide what to generate this cycle
        roll = random.random()

        if roll < SYNDICATE_BURST_CHANCE:
            # Full syndicate burst (all 3 layers)
            burst_type = "FULL SYNDICATE CYCLE"
            transactions_to_insert = generate_full_syndicate_burst(_active_boss)

        elif roll < FRAUD_INJECTION_RATE:
            # Single fraud event
            fraud_roll = random.random()
            if fraud_roll < 0.35:
                transactions_to_insert = generate_smurfing_burst(_active_boss)
                burst_type = "smurfing"
            elif fraud_roll < 0.60:
                transactions_to_insert = generate_layering_chain(_active_boss)
                burst_type = "layering"
            elif fraud_roll < 0.82:
                transactions_to_insert = generate_integration_transfer(_active_boss)
                burst_type = "integration"
            else:
                transactions_to_insert = generate_hawala_transaction()
                burst_type = "hawala"
        else:
            # Normal transaction
            transactions_to_insert = [generate_normal_transaction()]

        # Insert into database
        insert_transactions(conn, transactions_to_insert)

        # Print status
        for i, txn in enumerate(transactions_to_insert):
            is_fraud = txn["Fraud_Label"] != "NORMAL"
            print_status(txn, is_fraud, burst_type if i == 0 else None)
            if is_fraud:
                fraud_sent += 1
            total_sent += 1

        # Progress summary every 50 transactions
        if total_sent % 50 == 0:
            print(f"\n  ── Progress: {total_sent:,} sent | {fraud_sent:,} fraud "
                  f"({fraud_sent/total_sent*100:.1f}%) | Boss: {_active_boss} ──\n")

        time.sleep(delay)


if __name__ == "__main__":
    run()