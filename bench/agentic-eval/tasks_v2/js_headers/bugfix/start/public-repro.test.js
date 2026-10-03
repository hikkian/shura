const test=require('node:test'),assert=require('node:assert/strict'); const { parseHeaders }=require('./headers');
test('reported behavior',()=>{ assert.deepEqual(parseHeaders(['X-Name:  Alpha','\tbeta','x-name: Gamma','Accept: text/plain','']),{'x-name':['Alpha beta','Gamma'],accept:'text/plain'}) });
