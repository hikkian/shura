const test=require('node:test'),assert=require('node:assert/strict');const {parseHeaders}=require('../headers');
test('folds and combines case-insensitive names',()=>{assert.deepEqual(parseHeaders(['X-Name:  Alpha','\tbeta','x-name: Gamma','Accept: text/plain','']),{'x-name':['Alpha beta','Gamma'],accept:'text/plain'});});
test('rejects malformed lines',()=>{for(const a of [[' orphan'],['bad'],[' : x']])assert.throws(()=>parseHeaders(a),TypeError);});

test('supporting project files are completed',()=>{const fs=require('node:fs');const readme=fs.readFileSync('README.md','utf8'),design=fs.readFileSync('docs/design.md','utf8');assert.ok(readme.includes('headers'));assert.ok(!readme.includes('Starter project'));assert.ok(design.includes('Validation is explicit'));});

test('public package is split',()=>{const fs=require('node:fs');assert.ok(fs.existsSync('headers/index.js'));assert.ok(fs.existsSync('headers/core.js'));assert.ok(!fs.existsSync('headers.js'));});
