const test=require('node:test'),assert=require('node:assert/strict');const {replay}=require('../journal');
test('deduplicates and applies ordered operations',()=>{const ev=[{seq:1,op:'set',key:'x',value:2},{seq:1,op:'set',key:'x',value:9},{seq:2,op:'add',key:'x',value:3},{seq:3,op:'delete',key:'z'}];assert.deepEqual(replay(ev,{z:1}),{x:5});});
test('does not mutate and rejects bad events',()=>{const base={a:1};assert.throws(()=>replay([{seq:2,op:'set',key:'a',value:2},{seq:1,op:'set',key:'b',value:3}],base),RangeError);assert.deepEqual(base,{a:1});assert.throws(()=>replay([{seq:1,op:'wat',key:'x'}]),RangeError);});

test('supporting project files are completed',()=>{const fs=require('node:fs');const readme=fs.readFileSync('README.md','utf8'),design=fs.readFileSync('docs/design.md','utf8');assert.ok(readme.includes('journal'));assert.ok(!readme.includes('Starter project'));assert.ok(design.includes('Validation is explicit'));});

test('public package is split',()=>{const fs=require('node:fs');assert.ok(fs.existsSync('journal/index.js'));assert.ok(fs.existsSync('journal/core.js'));assert.ok(!fs.existsSync('journal.js'));});
