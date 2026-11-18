
import sqlite3
from datetime import datetime, date, timedelta

DB = "medremind.db"


def connect():
    return sqlite3.connect(DB)


def initialize():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medicines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            strength TEXT,
            quantity INTEGER NOT NULL,
            daily_consumption REAL DEFAULT 0,
            expiry_date TEXT NOT NULL,
            purchase_date TEXT,
            reorder_level INTEGER DEFAULT 5,
            member_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("Inventory database initialized!")


def add_medicine():
    name = input("Medicine name: ").strip()
    strength = input("Strength (e.g. 500mg): ").strip()
    quantity = int(input("Current quantity: "))
    daily = float(input("Daily consumption (units): "))
    expiry = input("Expiry date (YYYY-MM-DD): ").strip()
    purchase = input("Purchase date (YYYY-MM-DD): ").strip()
    reorder = int(input("Reorder threshold: "))
    member = input("Household member: ").strip()

    datetime.strptime(expiry, "%Y-%m-%d")

    if purchase:
        datetime.strptime(purchase, "%Y-%m-%d")

    if quantity < 0 or daily < 0 or reorder < 0:
        raise ValueError("Quantity, consumption and reorder threshold cannot be negative.")

    conn = connect()
    conn.execute("""
        INSERT INTO medicines
        (name, strength, quantity, daily_consumption,
         expiry_date, purchase_date, reorder_level, member_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, strength, quantity, daily, expiry,
          purchase or None, reorder, member))

    conn.commit()
    conn.close()
    print("Medicine added successfully!")


def view_inventory():
    conn = connect()
    rows = conn.execute("SELECT * FROM medicines").fetchall()
    conn.close()

    print("\nMEDICINE INVENTORY")
    print("-" * 90)

    for row in rows:
        print(
            f"ID: {row[0]} | {row[1]} {row[2]} | "
            f"Qty: {row[3]} | Expiry: {row[5]} | Member: {row[8]}"
        )


def expiry_alerts():
    conn = connect()
    rows = conn.execute(
        "SELECT name, strength, expiry_date FROM medicines"
    ).fetchall()
    conn.close()

    today = date.today()

    print("\nEXPIRY ALERTS")

    for name, strength, expiry in rows:
        expiry_day = datetime.strptime(expiry, "%Y-%m-%d").date()
        days = (expiry_day - today).days

        if days < 0:
            print(f"EXPIRED: {name} {strength} ({abs(days)} days ago)")
        elif days <= 30:
            print(f"EXPIRING SOON: {name} {strength} ({days} days left)")


def reorder_alerts():
    conn = connect()
    rows = conn.execute("""
        SELECT id, name, strength, quantity,
               daily_consumption, reorder_level
        FROM medicines
    """).fetchall()
    conn.close()

    today = date.today()
    lead_time = 3

    print("\nSMART REORDER PREDICTION")
    print("-" * 60)

    for medicine_id, name, strength, quantity, daily, threshold in rows:

        print(f"\n{medicine_id}. {name} {strength}")

        if quantity <= 0:
            print("STATUS: OUT OF STOCK")
            continue

        if daily <= 0:
            if quantity <= threshold:
                print("STATUS: LOW STOCK - REORDER NOW")
            else:
                print("Daily consumption not entered.")
            continue

        days_left = quantity / daily
        stockout_date = today + timedelta(days=days_left)
        reorder_date = stockout_date - timedelta(days=lead_time)

        print(f"Current quantity: {quantity}")
        print(f"Estimated stock remaining: {days_left:.1f} days")
        print(f"Expected stock-out: {stockout_date}")

        if quantity <= threshold or days_left <= lead_time:
            print("STATUS: REORDER NOW")
        else:
            print(f"Recommended reorder date: {reorder_date}")

def delete_medicine():
    medicine_id = int(input("Enter medicine ID to delete: "))

    conn = connect()
    cursor = conn.execute(
        "DELETE FROM medicines WHERE id = ?", (medicine_id,)
    )
    conn.commit()
    conn.close()

    print("Medicine deleted." if cursor.rowcount else "Medicine ID not found.")

def update_quantity():
    medicine_id = int(input("Enter medicine ID: "))
    quantity = int(input("Enter updated quantity: "))

    if quantity < 0:
        print("Quantity cannot be negative.")
        return

    conn = connect()
    cursor = conn.execute(
        "UPDATE medicines SET quantity = ? WHERE id = ?",
        (quantity, medicine_id)
    )
    conn.commit()
    conn.close()

    print("Quantity updated." if cursor.rowcount else "Medicine ID not found.")

def menu():
    initialize()

    while True:
        print("""
========= MEDREMIND =========
1. Add Medicine
2. View Inventory
3. Expiry Alerts
4. Reorder Prediction
5. Delete Medicine
6. Update Medicine Quantity
7. Exit
""")

        choice = input("Choose option: ")

        try:
            if choice == "1":
                add_medicine()
            elif choice == "2":
                view_inventory()
            elif choice == "3":
                expiry_alerts()
            elif choice == "4":
                reorder_alerts()
            elif choice == "5":
                delete_medicine()
            elif choice == "6":
                update_quantity()
            elif choice == "7":
                break
            else:
                print("Invalid choice.")
        except (ValueError, sqlite3.Error) as e:
            print("Error:", e)


if __name__ == "__main__":
    menu()
