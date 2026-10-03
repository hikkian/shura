const test=require('node:test'),assert=require('node:assert/strict');const {enabled}=require('../rollout');
test('boundaries and stable cohort',()=>{assert.equal(enabled('new-ui','u1',0),false);assert.equal(enabled('new-ui','u1',100),true);assert.equal(enabled('new-ui','user-17',37,'release'),true);});
test('validates inputs',()=>{for(const a of [['','u',50],['x','',50],['x','u',-1],['x','u',101],['x','u',NaN]])assert.throws(()=>enabled(...a),RangeError);});

test('supporting project files are completed',()=>{const fs=require('node:fs');const readme=fs.readFileSync('README.md','utf8'),design=fs.readFileSync('docs/design.md','utf8');assert.ok(readme.includes('rollout'));assert.ok(!readme.includes('Starter project'));assert.ok(design.includes('Validation is explicit'));});
