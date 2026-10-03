const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../headers');
test('documents an important edge case',()=>{ const { parseHeaders }=api; assert.deepEqual(parseHeaders(['X-Name:  Alpha','\tbeta','x-name: Gamma','Accept: text/plain','']),{'x-name':['Alpha beta','Gamma'],accept:'text/plain'}) });
