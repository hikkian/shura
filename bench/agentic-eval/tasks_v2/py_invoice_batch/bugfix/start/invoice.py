def invoice(lines,tax_bps=0):
    if isinstance(tax_bps,bool) or not isinstance(tax_bps,int) or tax_bps<0: raise ValueError("tax rate")
    subtotal=0
    for row in lines:
        q=row.get("quantity"); cents=row.get("unit_cents")
        if isinstance(q,bool) or not isinstance(q,int) or q<=0 or isinstance(cents,bool) or not isinstance(cents,int) or cents<0: raise ValueError("line")
        subtotal+=q*cents
    tax=(subtotal*tax_bps)//10000
    return {"subtotal":subtotal,"tax":tax,"total":subtotal+tax}
