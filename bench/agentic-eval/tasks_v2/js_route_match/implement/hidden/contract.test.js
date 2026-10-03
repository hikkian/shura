const test=require('node:test'),assert=require('node:assert/strict');const {matchRoute}=require('../routes');
test('matches and decodes',()=>{assert.deepEqual(matchRoute('/users/:id/files/:name','/users/42/files/a%20b'),{id:'42',name:'a b'});assert.deepEqual(matchRoute('/','/'),{});assert.equal(matchRoute('/users/:id','/users/1/edit'),null);assert.equal(matchRoute('/users/:id','/users/%FF'),null);});
test('rejects invalid patterns',()=>{assert.throws(()=>matchRoute('/x/:/y','/x/y/z'),RangeError);assert.throws(()=>matchRoute('/:id/:id','/a/b'),RangeError);});

test('supporting project files are completed',()=>{const fs=require('node:fs');const readme=fs.readFileSync('README.md','utf8'),design=fs.readFileSync('docs/design.md','utf8');assert.ok(readme.includes('routes'));assert.ok(!readme.includes('Starter project'));assert.ok(design.includes('Validation is explicit'));});
