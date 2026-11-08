def sales_by_day(conn, day):
    cur = conn.execute("SELECT region, SUM(amount) FROM sales WHERE day = ? GROUP BY region", (day,))
    return cur.fetchall()
