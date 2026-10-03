const test=require('node:test'),assert=require('node:assert/strict');const {delayFor}=require('../retrying');
test('exponential cap and deterministic jitter',()=>{assert.deepEqual([1,2,3,4,5].map(n=>delayFor(n,1,5)),[1,2,4,5,5]);assert.equal(delayFor(3,1,20,.25,0),3);assert.equal(delayFor(3,1,20,.25,1),5);});
test('rejects invalid inputs',()=>{for(const a of [[0,1,4],[true,1,4],[1,-1,4],[1,2,1],[1,1,2,1.1],[1,1,2,0,NaN]])assert.throws(()=>delayFor(...a),RangeError);});
