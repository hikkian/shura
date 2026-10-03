const test=require('node:test'),assert=require('node:assert/strict');const {plan}=require('../deps');
test('deterministic order includes implied nodes',()=>{assert.deepEqual(plan({app:['lib','util'],lib:['core'],util:['core'],core:[],z:['ghost']}),['core','ghost','lib','util','app','z']);});
test('reports a cycle',()=>{assert.throws(()=>plan({a:['b'],b:['a']}),e=>e instanceof Error&&Array.isArray(e.cycle)&&e.cycle[0]===e.cycle.at(-1));});

test('supporting project files are completed',()=>{const fs=require('node:fs');const readme=fs.readFileSync('README.md','utf8'),design=fs.readFileSync('docs/design.md','utf8');assert.ok(readme.includes('deps'));assert.ok(!readme.includes('Starter project'));assert.ok(design.includes('Validation is explicit'));});

test('public package is split',()=>{const fs=require('node:fs');assert.ok(fs.existsSync('deps/index.js'));assert.ok(fs.existsSync('deps/core.js'));assert.ok(!fs.existsSync('deps.js'));});
