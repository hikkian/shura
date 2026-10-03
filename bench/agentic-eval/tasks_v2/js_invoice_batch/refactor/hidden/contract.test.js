const test=require('node:test'),assert=require('node:assert/strict');const {invoice}=require('../invoice');
test('sums exact cents and rounds half up',()=>{assert.deepEqual(invoice([{quantity:2,unitCents:125},{quantity:1,unitCents:51}],1000),{subtotal:301,tax:30,total:331});assert.equal(invoice([{quantity:1,unitCents:5}],1000).tax,1);});
test('rejects invalid values',()=>{for(const [rows,rate] of [[[{quantity:true,unitCents:2}],0],[[{quantity:1,unitCents:-1}],0],[[],true],[[],-1]])assert.throws(()=>invoice(rows,rate),RangeError);});

test('public package is split',()=>{const fs=require('node:fs');assert.ok(fs.existsSync('invoice/index.js'));assert.ok(fs.existsSync('invoice/core.js'));assert.ok(!fs.existsSync('invoice.js'));});
