const test=require('node:test'),assert=require('node:assert/strict');const {deepMerge}=require('../config-merge');
test('recurses and replaces arrays and null',()=>{const a={db:{host:'a',ports:[1,2]},x:4},b={db:{port:9,ports:[3]},x:null};assert.deepEqual(deepMerge(a,b),{db:{host:'a',port:9,ports:[3]},x:null});assert.deepEqual(a.db.ports,[1,2]);});
test('returns fresh values and validates roots',()=>{const a={x:{y:1}},o=deepMerge(a,{});o.x.y=3;assert.equal(a.x.y,1);assert.throws(()=>deepMerge([],{}),TypeError);});

test('supporting project files are completed',()=>{const fs=require('node:fs');const readme=fs.readFileSync('README.md','utf8'),design=fs.readFileSync('docs/design.md','utf8');assert.ok(readme.includes('config-merge'));assert.ok(!readme.includes('Starter project'));assert.ok(design.includes('Validation is explicit'));});

test('public package is split',()=>{const fs=require('node:fs');assert.ok(fs.existsSync('config-merge/index.js'));assert.ok(fs.existsSync('config-merge/core.js'));assert.ok(!fs.existsSync('config-merge.js'));});
