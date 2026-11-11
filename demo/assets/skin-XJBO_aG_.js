import{A as e,C as t,D as n,E as r,F as i,M as a,N as o,O as s,P as c,S as l,T as u,_ as d,a as f,b as p,c as m,d as h,f as g,g as _,h as v,i as ee,j as te,l as y,m as b,n as ne,o as x,p as re,r as ie,s as ae,u as S,v as oe,w as C,x as se,y as ce}from"./index-CmMYTyew.js";var le=1/60,ue=1/240,de=.8/ue,fe=.02,w=.01,pe=.002,me=2,he=.25,T=1e3,E=e=>String(Number(e.toPrecision(7))),D=e=>`${E(e.x)} ${E(e.y)} ${E(e.z)}`,ge=e=>`${E(e.w)} ${E(e.x)} ${E(e.y)} ${E(e.z)}`;function O(e){return!!e&&`shape`in e&&`colliderMatrix`in e}var _e=class e{constructor(e,t,n,r,a){this.mujoco=e,this.model=t,this.data=n,this.anchors=r,this.links=a,this.position=new i,this.quaternion=new s,this.scale=new i,this.target=new s,this.turn=new s,this.ball=new s,this.spin=new i}get jointCount(){return this.links.length}static build(t,n,r){let{xml:a,anchors:o,links:c}=e.describe(n,r);if(!c.length)return null;let l=t.MjModel.from_xml_string(a),u=new t.MjData(l),d=e=>l.jnt_qposadr[l.body_jntadr[e]];return new e(t,l,u,o.map(({node:e,body:t})=>({node:e,body:t,qposAdr:d(t),dofAdr:l.jnt_dofadr[l.body_jntadr[t]],fromPos:new i,fromQuat:new s,toPos:new i,toQuat:new s})),c.map(({bone:e,body:t})=>({bone:e,qposAdr:d(t),rest:e.quaternion.clone()})))}static describe(t,n){n.updateMatrixWorld(!0);let r=new C().copy(n.matrixWorld).invert(),a=e=>{let t=new i,n=new s;return new C().multiplyMatrices(r,e.matrixWorld).decompose(t,n,new i),{pos:t,quat:n}},o=(e,t)=>e.localToWorld(t.clone()).applyMatrix4(r),c=new Map;for(let e of t.joints)!O(e.bone)&&!c.has(e.bone)&&c.set(e.bone,e);let l=new Map,u=[],d=e=>{let t=l.get(e);if(t)return t;let n=c.get(e);return t={node:e,joint:n,children:[],geoms:[],...a(e)},l.set(e,t),n&&e.parent?d(e.parent).children.push(t):u.push(t),t},f=new Map;for(let e of c.values())for(let t of e.colliderGroups)f.has(t)||f.set(t,1<<f.size%30);let p=new Map;for(let[e,t]of f)for(let n of e.colliders)p.set(n,(p.get(n)??0)|t);let m=0;for(let[t,n]of c){let r=d(t);r.tail=o(t,n.initialLocalChildPosition).sub(r.pos).applyQuaternion(r.quat.clone().invert()),r.mass=fe*Math.max(r.tail.length(),w),m=Math.max(m,e.gravityOf(n))}for(let[e,t]of p){let n=e.parent,r=e.shape;if(!n||r.inside||r.type!==`sphere`&&r.type!==`capsule`)continue;let a=d(n),s=e=>o(n,e).sub(a.pos).applyQuaternion(a.quat.clone().invert()),c=E(Math.max(r.radius??0,pe)),l=s(r.offset??new i),u=`contype="${t}" conaffinity="0"`;a.geoms.push(r.type===`sphere`?`<geom type="sphere" pos="${D(l)}" size="${c}" ${u}/>`:`<geom type="capsule" fromto="${D(l)} ${D(s(r.tail??new i))}" size="${c}" ${u}/>`)}let h=[],g=[],_=1,v=(t,n)=>{let r=_++,i=t.joint,a=t.pos,o=t.quat;if(n){let e=n.quat.clone().invert();a=t.pos.clone().sub(n.pos).applyQuaternion(e),o=e.multiply(t.quat)}let s=[];i?(g.push({bone:t.node,body:r}),s.push(...e.jointXml(t,i,f))):(h.push({node:t.node,body:r}),s.push(`<freejoint/>`,`<inertial pos="0 0 0" mass="${T}" diaginertia="${T} ${T} ${T}"/>`)),s.push(...t.geoms);for(let e of t.children)s.push(v(e,t));let c=m>0?` gravcomp="${E(i?1-e.gravityOf(i)/m:1)}"`:``;return`<body pos="${D(a)}" quat="${ge(o)}"${c}>\n${s.join(`
`)}\n</body>`},ee=u.map(e=>v(e,null)).join(`
`),te=p.size>0;return{xml:`<mujoco model="skin springs">
<option timestep="${E(ue)}" gravity="0 ${E(-m)} 0" integrator="implicitfast" jacobian="sparse">
<flag filterparent="disable"${te?``:` contact="disable"`}/>
</option>
<default><geom condim="1" solref="${E(2*ue)} 1" solimp="0.95 0.99 0.001"/></default>
<worldbody>
${ee}
</worldbody>
</mujoco>`,anchors:h,links:g}}static gravityOf(e){let t=Math.max(0,-e.settings.gravityDir.y);return e.settings.gravityPower*t/le}static jointXml(t,n,r){let i=t.tail,a=Math.max(i.length(),w),o=e.carriedInertia(t),s=he*o,c=o+s,l=Math.min(Math.sqrt(n.settings.stiffness/(le*a)),de),u=Math.max(-Math.log(1-Math.min(Math.max(n.settings.dragForce,0),.95))/le,me),d=0;for(let e of n.colliderGroups)d|=r.get(e)??0;let f=t.mass,p=.1*f*a*a;return[`<joint type="ball" stiffness="${E(c*l*l)}" damping="${E(c*u)}" armature="${E(s)}"/>`,`<inertial pos="${D(i)}" mass="${E(f)}" diaginertia="${E(p)} ${E(p)} ${E(p)}"/>`,`<geom type="sphere" pos="${D(i)}" size="${E(Math.max(n.settings.hitRadius,pe))}" contype="0" conaffinity="${d}"/>`]}static carriedInertia(e){let t=e.pos,n=0,r=e=>{if(e.tail&&e.mass){let r=e.tail.clone().applyQuaternion(e.quat).add(e.pos),i=Math.max(e.tail.length(),w);n+=e.mass*r.distanceToSquared(t)+.1*e.mass*i*i}for(let t of e.children)t.joint&&r(t)};return r(e),Math.max(n,1e-9)}reset(){this.mujoco.mj_resetData(this.model,this.data);let e=this.data.qpos;for(let t of this.anchors)this.readPose(t.node,t.toPos,t.toQuat),t.fromPos.copy(t.toPos),t.fromQuat.copy(t.toQuat),e.set([t.toPos.x,t.toPos.y,t.toPos.z],t.qposAdr),e.set([t.toQuat.w,t.toQuat.x,t.toQuat.y,t.toQuat.z],t.qposAdr+3);this.writeBones()}step(e){for(let e of this.anchors)this.readPose(e.node,e.toPos,e.toQuat);let t=Math.max(1,Math.ceil(e/ue-1e-6)),n=e/t;this.model.opt.timestep=n;for(let e=1;e<=t;e++){let r=e/t;for(let e of this.anchors)this.drive(e,r,n);this.mujoco.mj_step(this.model,this.data)}for(let e of this.anchors)e.fromPos.copy(e.toPos),e.fromQuat.copy(e.toQuat);this.writeBones()}drive(e,t,n){let r=this.data.qpos,i=this.data.qvel,a=this.data.xfrc_applied,o=this.position.lerpVectors(e.fromPos,e.toPos,t),s=this.quaternion.set(r[e.qposAdr+4],r[e.qposAdr+5],r[e.qposAdr+6],r[e.qposAdr+3]),c=e.body*6;for(let t=0;t<3;t++){let s=(o.getComponent(t)-r[e.qposAdr+t])/n;a[c+t]=T*(s-i[e.dofAdr+t])/n}this.target.slerpQuaternions(e.fromQuat,e.toQuat,t);let l=this.turn.copy(s).invert().multiply(this.target);l.w<0&&l.set(-l.x,-l.y,-l.z,-l.w);let u=Math.hypot(l.x,l.y,l.z),d=2*Math.atan2(u,l.w),f=u>1e-12?this.spin.set(l.x,l.y,l.z).multiplyScalar(d/u/n):this.spin.set(0,0,0);f.set(T*(f.x-i[e.dofAdr+3])/n,T*(f.y-i[e.dofAdr+4])/n,T*(f.z-i[e.dofAdr+5])/n).applyQuaternion(s),a[c+3]=f.x,a[c+4]=f.y,a[c+5]=f.z}readPose(e,t,n){e.matrixWorld.decompose(t,n,this.scale)}writeBones(){let e=this.data.qpos;for(let t of this.links){let n=t.qposAdr;this.ball.set(e[n+1],e[n+2],e[n+3],e[n]),t.bone.quaternion.copy(t.rest).multiply(this.ball),t.bone.updateMatrix()}}dispose(){this.data.delete(),this.model.delete()}},ve=new s().setFromAxisAngle(new i(0,1,0),Math.PI/2);function ye(e){return e===`0`?new s(0,1,0,0):new s}function be(e,t,n){return n.set(e[t*4+1],e[t*4+3],-e[t*4+2],e[t*4])}function xe(e,t){let n=[],r=new s;for(let i=0;i<e.nbody;i++){let a=new s;if(n.push(a),i===0)continue;let o=e.body_jntadr[i];e.body_jntnum[i]>0&&e.jnt_type[o]===t||a.copy(n[e.body_parentid[i]]).multiply(be(e.body_quat,i,r))}return n}function Se(e,t,n){let r=[];for(let t=n;t>0;t=e.body_parentid[t])r.unshift(t);let a=new i,o=new i;for(let n of r){let r=e.body_pos;o.set(r[n*3],r[n*3+2],-r[n*3+1]),a.add(o.applyQuaternion(t[e.body_parentid[n]]))}return a.y}function Ce(e,t,n,r){let i=[...e].sort((e,t)=>e.ancestors.length-t.ancestors.length),a=new Map(i.map((e,t)=>[e.name,t]));return{facing:ve.clone(),frame:r.clone(),bones:i.map(e=>{let r=e.ancestors.find(e=>a.has(e)),i=n[e.name];return{name:e.name,bodyId:e.bodyId,parent:r===void 0?null:a.get(r),restInverse:t[e.bodyId].clone().invert(),offset:i?new s(i[0],i[1],i[2],i[3]).normalize():new s}})}}var we=new s,Te=new s,Ee=new s,De=new s;function Oe(e,t,n,r){we.copy(e.facing).invert(),Te.copy(e.frame).invert(),e.bones.forEach((i,a)=>{n[a].copy(t(i.bodyId)).multiply(i.restInverse).premultiply(we).multiply(e.facing).multiply(i.offset);let o=r[a].copy(n[a]);i.parent!==null&&o.premultiply(Ee.copy(n[i.parent]).invert()),o.premultiply(Te).multiply(e.frame)})}function ke(e,t,n,r,i){return De.copy(e.facing).multiply(e.frame).invert(),i.set(t.x,r+(t.y-r)*n,t.z).applyQuaternion(De)}var k=new i,A=new i,j=new i,M=new i,Ae=new i,N=new i,je=new i,Me=new i,P=new i,Ne=new i,Pe=new s,Fe=new s,Ie=new s,Le=new s,Re=new s,ze=new s,Be=new s,Ve=new s,He=1e-4;function Ue(e,t,n,r,i,a){e.getWorldPosition(k),t.getWorldPosition(A),n.getWorldPosition(j),e.getWorldQuaternion(Pe),t.getWorldQuaternion(Fe),n.getWorldQuaternion(Ie);let o=k.distanceTo(A),s=A.distanceTo(j),c=l.clamp(k.distanceTo(r),Math.abs(o-s)+He,o+s-He);M.subVectors(j,k),Ae.subVectors(A,k),N.subVectors(r,k);let u=e=>l.clamp(e,-1,1),d=M.angleTo(Ae),f=je.subVectors(k,A).angleTo(Me.subVectors(j,A)),p=Math.acos(u((s*s-o*o-c*c)/(-2*o*c))),m=Math.acos(u((c*c-o*o-s*s)/(-2*o*s)));P.crossVectors(M,Ae),P.lengthSq()>1e-12?i.copy(P.normalize()):P.copy(i),ze.setFromAxisAngle(P,p-d),Be.setFromAxisAngle(P,m-f),Ne.crossVectors(M,N),Ne.lengthSq()>1e-12?Ve.setFromAxisAngle(Ne.normalize(),M.angleTo(N)):Ve.identity(),Pe.premultiply(ze).premultiply(Ve),Fe.premultiply(ze).premultiply(Be).premultiply(Ve),e.parent.getWorldQuaternion(Le),e.quaternion.copy(Le.invert().multiply(Pe)),t.quaternion.copy(Re.copy(Pe).invert().multiply(Fe)),n.quaternion.copy(Fe.invert().multiply(a??Ie)),e.updateMatrixWorld(!0)}function We(e,t,n,r){e.getWorldPosition(k),t.getWorldPosition(A),n.getWorldPosition(j);let i=k.distanceTo(A)+A.distanceTo(j)-He;N.subVectors(r,k);let a=N.lengthSq()-i*i,o=-N.y,s=o*o-a;return a<=0||s<0?0:Math.max(0,o-Math.sqrt(s))}var Ge=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())}),F=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())}),Ke=class extends n{constructor(e){super(),this.weight=0,this.isBinary=!1,this.overrideBlink=`none`,this.overrideLookAt=`none`,this.overrideMouth=`none`,this._binds=[],this.name=`VRMExpression_${e}`,this.expressionName=e,this.type=`VRMExpression`,this.visible=!1}get binds(){return this._binds}get overrideBlinkAmount(){return this.overrideBlink===`block`?+(0<this.outputWeight):this.overrideBlink===`blend`?this.outputWeight:0}get overrideLookAtAmount(){return this.overrideLookAt===`block`?+(0<this.outputWeight):this.overrideLookAt===`blend`?this.outputWeight:0}get overrideMouthAmount(){return this.overrideMouth===`block`?+(0<this.outputWeight):this.overrideMouth===`blend`?this.outputWeight:0}get outputWeight(){return this.isBinary?+(this.weight>.5):this.weight}addBind(e){this._binds.push(e)}deleteBind(e){let t=this._binds.indexOf(e);t>=0&&this._binds.splice(t,1)}applyWeight(e){let t=this.outputWeight;t*=e?.multiplier??1,this.isBinary&&t<1&&(t=0),this._binds.forEach(e=>e.applyWeight(t))}clearAppliedWeight(){this._binds.forEach(e=>e.clearAppliedWeight())}};function qe(e,t,n){let r=e.parser.json,i=r.nodes?.[t];if(i==null)return null;let a=i.mesh;if(a==null)return null;let o=r.meshes?.[a];if(o==null)return null;let s=o.primitives.length,c=[];return n.traverse(e=>{c.length<s&&e.isMesh&&c.push(e)}),c}function Je(e,t){return F(this,null,function*(){return qe(e,t,yield e.parser.getDependency(`node`,t))})}function Ye(e){return F(this,null,function*(){let t=yield e.parser.getDependencies(`node`),n=new Map;return t.forEach((t,r)=>{let i=qe(e,r,t);i!=null&&n.set(r,i)}),n})}var Xe={Aa:`aa`,Ih:`ih`,Ou:`ou`,Ee:`ee`,Oh:`oh`,Blink:`blink`,Happy:`happy`,Angry:`angry`,Sad:`sad`,Relaxed:`relaxed`,LookUp:`lookUp`,Surprised:`surprised`,LookDown:`lookDown`,LookLeft:`lookLeft`,LookRight:`lookRight`,BlinkLeft:`blinkLeft`,BlinkRight:`blinkRight`,Neutral:`neutral`};function Ze(e){return Math.max(Math.min(e,1),0)}var Qe=class e{constructor(){this.blinkExpressionNames=[`blink`,`blinkLeft`,`blinkRight`],this.lookAtExpressionNames=[`lookLeft`,`lookRight`,`lookUp`,`lookDown`],this.mouthExpressionNames=[`aa`,`ee`,`ih`,`oh`,`ou`],this._expressions=[],this._expressionMap={}}get expressions(){return this._expressions.concat()}get expressionMap(){return Object.assign({},this._expressionMap)}get presetExpressionMap(){let e={},t=new Set(Object.values(Xe));return Object.entries(this._expressionMap).forEach(([n,r])=>{t.has(n)&&(e[n]=r)}),e}get customExpressionMap(){let e={},t=new Set(Object.values(Xe));return Object.entries(this._expressionMap).forEach(([n,r])=>{t.has(n)||(e[n]=r)}),e}copy(e){return this._expressions.concat().forEach(e=>{this.unregisterExpression(e)}),e._expressions.forEach(e=>{this.registerExpression(e)}),this.blinkExpressionNames=e.blinkExpressionNames.concat(),this.lookAtExpressionNames=e.lookAtExpressionNames.concat(),this.mouthExpressionNames=e.mouthExpressionNames.concat(),this}clone(){return new e().copy(this)}getExpression(e){return this._expressionMap[e]??null}registerExpression(e){this._expressions.push(e),this._expressionMap[e.expressionName]=e}unregisterExpression(e){let t=this._expressions.indexOf(e);this._expressions.splice(t,1),delete this._expressionMap[e.expressionName]}getValue(e){return this.getExpression(e)?.weight??null}setValue(e,t){let n=this.getExpression(e);n&&(n.weight=Ze(t))}resetValues(){this._expressions.forEach(e=>{e.weight=0})}getExpressionTrackName(e){let t=this.getExpression(e);return t?`${t.name}.weight`:null}update(){let e=this._calculateWeightMultipliers();this._expressions.forEach(e=>{e.clearAppliedWeight()}),this._expressions.forEach(t=>{let n=1,r=t.expressionName;this.blinkExpressionNames.indexOf(r)!==-1&&(n*=e.blink),this.lookAtExpressionNames.indexOf(r)!==-1&&(n*=e.lookAt),this.mouthExpressionNames.indexOf(r)!==-1&&(n*=e.mouth),t.applyWeight({multiplier:n})})}_calculateWeightMultipliers(){let e=1,t=1,n=1;return this._expressions.forEach(r=>{e-=r.overrideBlinkAmount,t-=r.overrideLookAtAmount,n-=r.overrideMouthAmount}),e=Math.max(0,e),t=Math.max(0,t),n=Math.max(0,n),{blink:e,lookAt:t,mouth:n}}},$e={Color:`color`,EmissionColor:`emissionColor`,ShadeColor:`shadeColor`,MatcapColor:`matcapColor`,RimColor:`rimColor`,OutlineColor:`outlineColor`},et={_Color:$e.Color,_EmissionColor:$e.EmissionColor,_ShadeColor:$e.ShadeColor,_RimColor:$e.RimColor,_OutlineColor:$e.OutlineColor},tt=new S,nt=class e{constructor({material:e,type:t,targetValue:n,targetAlpha:r}){this.material=e,this.type=t,this.targetValue=n,this.targetAlpha=r??1;let i=this._initColorBindState(),a=this._initAlphaBindState();this._state={color:i,alpha:a}}applyWeight(e){let{color:t,alpha:n}=this._state;if(t!=null){let{propertyName:n,deltaValue:r}=t;this.material[n]?.add(tt.copy(r).multiplyScalar(e))}if(n!=null){let{propertyName:t,deltaValue:r}=n;this.material[t]!=null&&(this.material[t]+=r*e)}}clearAppliedWeight(){let{color:e,alpha:t}=this._state;if(e!=null){let{propertyName:t,initialValue:n}=e;this.material[t]?.copy(n)}if(t!=null){let{propertyName:e,initialValue:n}=t;this.material[e]!=null&&(this.material[e]=n)}}_initColorBindState(){let{material:e,type:t,targetValue:n}=this,r=this._getPropertyNameMap()?.[t]?.[0]??null;if(r==null)return null;let i=e[r].clone();return{propertyName:r,initialValue:i,deltaValue:new S(n.r-i.r,n.g-i.g,n.b-i.b)}}_initAlphaBindState(){let{material:e,type:t,targetAlpha:n}=this,r=this._getPropertyNameMap()?.[t]?.[1]??null;if(r==null&&n!==1||r==null)return null;let i=e[r];return{propertyName:r,initialValue:i,deltaValue:n-i}}_getPropertyNameMap(){return Object.entries(e._propertyNameMapMap).find(([e])=>this.material[e]===!0)?.[1]??null}};nt._propertyNameMapMap={isMeshStandardMaterial:{color:[`color`,`opacity`],emissionColor:[`emissive`,null]},isMeshBasicMaterial:{color:[`color`,`opacity`]},isMToonMaterial:{color:[`color`,`opacity`],emissionColor:[`emissive`,null],outlineColor:[`outlineColorFactor`,null],matcapColor:[`matcapFactor`,null],rimColor:[`parametricRimColorFactor`,null],shadeColor:[`shadeColorFactor`,null]}};var rt=nt,it=class{constructor({primitives:e,index:t,weight:n}){this.primitives=e,this.index=t,this.weight=n}applyWeight(e){this.primitives.forEach(t=>{t.morphTargetInfluences?.[this.index]!=null&&(t.morphTargetInfluences[this.index]+=this.weight*e)})}clearAppliedWeight(){this.primitives.forEach(e=>{e.morphTargetInfluences?.[this.index]!=null&&(e.morphTargetInfluences[this.index]=0)})}},at=new c,ot=class e{constructor({material:t,scale:n,offset:r}){this.material=t,this.scale=n,this.offset=r;let i=Object.entries(e._propertyNamesMap).find(([e])=>t[e]===!0)?.[1];i==null?this._properties=[]:(this._properties=[],i.forEach(e=>{let i=t[e]?.clone();if(!i)return null;t[e]=i;let a=i.offset.clone(),o=i.repeat.clone(),s=r.clone().sub(a),c=n.clone().sub(o);this._properties.push({name:e,initialOffset:a,deltaOffset:s,initialScale:o,deltaScale:c})}))}applyWeight(e){this._properties.forEach(t=>{let n=this.material[t.name];n!==void 0&&(n.offset.add(at.copy(t.deltaOffset).multiplyScalar(e)),n.repeat.add(at.copy(t.deltaScale).multiplyScalar(e)))})}clearAppliedWeight(){this._properties.forEach(e=>{let t=this.material[e.name];t!==void 0&&(t.offset.copy(e.initialOffset),t.repeat.copy(e.initialScale))})}};ot._propertyNamesMap={isMeshStandardMaterial:[`map`,`emissiveMap`,`bumpMap`,`normalMap`,`displacementMap`,`roughnessMap`,`metalnessMap`,`alphaMap`],isMeshBasicMaterial:[`map`,`specularMap`,`alphaMap`],isMToonMaterial:[`map`,`normalMap`,`emissiveMap`,`shadeMultiplyTexture`,`rimMultiplyTexture`,`outlineWidthMultiplyTexture`,`uvAnimationMaskTexture`]};var st=ot,ct=new Set([`1.0`,`1.0-beta`]),lt=class e{get name(){return`VRMExpressionLoaderPlugin`}constructor(e){this.parser=e}afterRoot(e){return F(this,null,function*(){e.userData.vrmExpressionManager=yield this._import(e)})}_import(e){return F(this,null,function*(){return(yield this._v1Import(e))||(yield this._v0Import(e))||null})}_v1Import(e){return F(this,null,function*(){let t=this.parser.json;if(t.extensionsUsed?.indexOf(`VRMC_vrm`)===-1)return null;let n=t.extensions?.VRMC_vrm;if(!n)return null;let r=n.specVersion;if(!ct.has(r))return null;let i=n.expressions;if(!i)return null;let a=new Set(Object.values(Xe)),o=new Map;i.preset!=null&&Object.entries(i.preset).forEach(([e,t])=>{t!=null&&a.has(e)&&o.set(e,t)}),i.custom!=null&&Object.entries(i.custom).forEach(([e,t])=>{a.has(e)||o.set(e,t)});let s=new Qe;return yield Promise.all(Array.from(o.entries()).map(t=>F(this,[t],function*([t,n]){var r,i,a;let o=new Ke(t);if(e.scene.add(o),o.isBinary=n.isBinary??!1,o.overrideBlink=n.overrideBlink??`none`,o.overrideLookAt=n.overrideLookAt??`none`,o.overrideMouth=n.overrideMouth??`none`,(r=n.morphTargetBinds)==null||r.forEach(t=>F(this,null,function*(){if(t.node===void 0||t.index===void 0)return;let n=yield Je(e,t.node),r=t.index;n.every(e=>Array.isArray(e.morphTargetInfluences)&&r<e.morphTargetInfluences.length)&&o.addBind(new it({primitives:n,index:r,weight:t.weight??1}))})),n.materialColorBinds||n.textureTransformBinds){let t=[];e.scene.traverse(e=>{let n=e.material;n&&(Array.isArray(n)?t.push(...n):t.push(n))}),(i=n.materialColorBinds)==null||i.forEach(e=>F(this,null,function*(){t.filter(t=>{let n=this.parser.associations.get(t)?.materials;return e.material===n}).forEach(t=>{o.addBind(new rt({material:t,type:e.type,targetValue:new S().fromArray(e.targetValue),targetAlpha:e.targetValue[3]}))})})),(a=n.textureTransformBinds)==null||a.forEach(e=>F(this,null,function*(){t.filter(t=>{let n=this.parser.associations.get(t)?.materials;return e.material===n}).forEach(t=>{o.addBind(new st({material:t,offset:new c().fromArray(e.offset??[0,0]),scale:new c().fromArray(e.scale??[1,1])}))})}))}s.registerExpression(o)}))),s})}_v0Import(t){return F(this,null,function*(){let n=this.parser.json,r=n.extensions?.VRM;if(!r)return null;let i=r.blendShapeMaster;if(!i)return null;let a=new Qe,o=i.blendShapeGroups;if(!o)return a;let s=new Set;return yield Promise.all(o.map(r=>F(this,null,function*(){let i=r.presetName,o=(i!=null&&e.v0v1PresetNameMap[i]||null)??r.name;if(o==null||s.has(o))return;s.add(o);let l=new Ke(o);t.scene.add(l),l.isBinary=r.isBinary??!1,r.binds&&r.binds.forEach(e=>F(this,null,function*(){var r;if(e.mesh===void 0||e.index===void 0)return;let i=[];if((r=n.nodes)==null||r.forEach((t,n)=>{t.mesh===e.mesh&&i.push(n)}),i.length===0)return;let a=e.index;yield Promise.all(i.map(n=>F(this,null,function*(){let r=yield Je(t,n);r.every(e=>Array.isArray(e.morphTargetInfluences)&&a<e.morphTargetInfluences.length)&&l.addBind(new it({primitives:r,index:a,weight:.01*(e.weight??100)}))})))}));let u=r.materialValues;u&&u.length!==0&&u.forEach(e=>{if(e.materialName===void 0||e.propertyName===void 0||e.targetValue===void 0)return;let n=[];t.scene.traverse(t=>{if(t.material){let r=t.material;Array.isArray(r)?n.push(...r.filter(t=>(t.name===e.materialName||t.name===e.materialName+` (Outline)`)&&n.indexOf(t)===-1)):r.name===e.materialName&&n.indexOf(r)===-1&&n.push(r)}});let r=e.propertyName;n.forEach(t=>{if(r===`_MainTex_ST`){let n=new c(e.targetValue[0],e.targetValue[1]),r=new c(e.targetValue[2],e.targetValue[3]);r.y=1-r.y-n.y,l.addBind(new st({material:t,scale:n,offset:r}));return}let n=et[r];n&&l.addBind(new rt({material:t,type:n,targetValue:new S().fromArray(e.targetValue),targetAlpha:e.targetValue[3]}))})}),a.registerExpression(l)}))),a})}};lt.v0v1PresetNameMap={a:`aa`,e:`ee`,i:`ih`,o:`oh`,u:`ou`,blink:`blink`,joy:`happy`,angry:`angry`,sorrow:`sad`,fun:`relaxed`,lookup:`lookUp`,lookdown:`lookDown`,lookleft:`lookLeft`,lookright:`lookRight`,blink_l:`blinkLeft`,blink_r:`blinkRight`,neutral:`neutral`};var ut=lt,dt=class e{constructor(t,n){this._firstPersonOnlyLayer=e.DEFAULT_FIRSTPERSON_ONLY_LAYER,this._thirdPersonOnlyLayer=e.DEFAULT_THIRDPERSON_ONLY_LAYER,this._initializedLayers=!1,this.humanoid=t,this.meshAnnotations=n}copy(e){if(this.humanoid!==e.humanoid)throw Error(`VRMFirstPerson: humanoid must be same in order to copy`);return this.meshAnnotations=e.meshAnnotations.map(e=>({meshes:e.meshes.concat(),type:e.type})),this}clone(){return new e(this.humanoid,this.meshAnnotations).copy(this)}get firstPersonOnlyLayer(){return this._firstPersonOnlyLayer}get thirdPersonOnlyLayer(){return this._thirdPersonOnlyLayer}setup({firstPersonOnlyLayer:t=e.DEFAULT_FIRSTPERSON_ONLY_LAYER,thirdPersonOnlyLayer:n=e.DEFAULT_THIRDPERSON_ONLY_LAYER}={}){this._initializedLayers||=(this._firstPersonOnlyLayer=t,this._thirdPersonOnlyLayer=n,this.meshAnnotations.forEach(e=>{e.meshes.forEach(t=>{e.type===`firstPersonOnly`?(t.layers.set(this._firstPersonOnlyLayer),t.traverse(e=>e.layers.set(this._firstPersonOnlyLayer))):e.type===`thirdPersonOnly`?(t.layers.set(this._thirdPersonOnlyLayer),t.traverse(e=>e.layers.set(this._thirdPersonOnlyLayer))):e.type===`auto`&&this._createHeadlessModel(t)})}),!0)}_excludeTriangles(e,t,n,r){let i=0;if(t!=null&&t.length>0)for(let a=0;a<e.length;a+=3){let o=e[a],s=e[a+1],c=e[a+2],l=t[o],u=n[o];if(l[0]>0&&r.includes(u[0])||l[1]>0&&r.includes(u[1])||l[2]>0&&r.includes(u[2])||l[3]>0&&r.includes(u[3]))continue;let d=t[s],f=n[s];if(d[0]>0&&r.includes(f[0])||d[1]>0&&r.includes(f[1])||d[2]>0&&r.includes(f[2])||d[3]>0&&r.includes(f[3]))continue;let p=t[c],m=n[c];p[0]>0&&r.includes(m[0])||p[1]>0&&r.includes(m[1])||p[2]>0&&r.includes(m[2])||p[3]>0&&r.includes(m[3])||(e[i++]=o,e[i++]=s,e[i++]=c)}return i}_createErasedMesh(e,t){let n=new a(e.geometry.clone(),e.material);n.name=`${e.name}(erase)`,n.frustumCulled=e.frustumCulled,n.layers.set(this._firstPersonOnlyLayer);let r=n.geometry,i=r.getAttribute(`skinIndex`),o=i instanceof re?[]:i.array,s=[];for(let e=0;e<o.length;e+=4)s.push([o[e],o[e+1],o[e+2],o[e+3]]);let c=r.getAttribute(`skinWeight`),l=c instanceof re?[]:c.array,u=[];for(let e=0;e<l.length;e+=4)u.push([l[e],l[e+1],l[e+2],l[e+3]]);let d=r.getIndex();if(!d)throw Error(`The geometry doesn't have an index buffer`);let f=Array.from(d.array),p=this._excludeTriangles(f,u,s,t),m=[];for(let e=0;e<p;e++)m[e]=f[e];return r.setIndex(m),e.onBeforeRender&&(n.onBeforeRender=e.onBeforeRender),n.bind(new te(e.skeleton.bones,e.skeleton.boneInverses),new C),n}_createHeadlessModelForSkinnedMesh(e,t){let n=[];if(t.skeleton.bones.forEach((e,t)=>{this._isEraseTarget(e)&&n.push(t)}),!n.length){t.layers.enable(this._thirdPersonOnlyLayer),t.layers.enable(this._firstPersonOnlyLayer);return}t.layers.set(this._thirdPersonOnlyLayer);let r=this._createErasedMesh(t,n);e.add(r)}_createHeadlessModel(e){if(e.type===`Group`){if(e.layers.set(this._thirdPersonOnlyLayer),this._isEraseTarget(e))e.traverse(e=>e.layers.set(this._thirdPersonOnlyLayer));else{let t=new b;t.name=`_headless_${e.name}`,t.layers.set(this._firstPersonOnlyLayer),e.parent.add(t),e.children.filter(e=>e.type===`SkinnedMesh`).forEach(e=>{let n=e;this._createHeadlessModelForSkinnedMesh(t,n)})}}else if(e.type===`SkinnedMesh`){let t=e;this._createHeadlessModelForSkinnedMesh(e.parent,t)}else this._isEraseTarget(e)&&(e.layers.set(this._thirdPersonOnlyLayer),e.traverse(e=>e.layers.set(this._thirdPersonOnlyLayer)))}_isEraseTarget(e){return e===this.humanoid.getRawBoneNode(`head`)?!0:e.parent?this._isEraseTarget(e.parent):!1}};dt.DEFAULT_FIRSTPERSON_ONLY_LAYER=9,dt.DEFAULT_THIRDPERSON_ONLY_LAYER=10;var ft=dt,pt=new Set([`1.0`,`1.0-beta`]),mt=class{get name(){return`VRMFirstPersonLoaderPlugin`}constructor(e){this.parser=e}afterRoot(e){return F(this,null,function*(){let t=e.userData.vrmHumanoid;if(t!==null){if(t===void 0)throw Error(`VRMFirstPersonLoaderPlugin: vrmHumanoid is undefined. VRMHumanoidLoaderPlugin have to be used first`);e.userData.vrmFirstPerson=yield this._import(e,t)}})}_import(e,t){return F(this,null,function*(){return t==null?null:(yield this._v1Import(e,t))||(yield this._v0Import(e,t))||null})}_v1Import(e,t){return F(this,null,function*(){let n=this.parser.json;if(n.extensionsUsed?.indexOf(`VRMC_vrm`)===-1)return null;let r=n.extensions?.VRMC_vrm;if(!r)return null;let i=r.specVersion;if(!pt.has(i))return null;let a=r.firstPerson,o=[],s=yield Ye(e);return Array.from(s.entries()).forEach(([e,t])=>{let n=(a?.meshAnnotations)?.find(t=>t.node===e);o.push({meshes:t,type:n?.type??`auto`})}),new ft(t,o)})}_v0Import(e,t){return F(this,null,function*(){let n=this.parser.json,r=n.extensions?.VRM;if(!r)return null;let i=r.firstPerson;if(!i)return null;let a=[],o=yield Ye(e);return Array.from(o.entries()).forEach(([e,t])=>{let r=n.nodes[e],o=i.meshAnnotations?i.meshAnnotations.find(e=>e.mesh===r.mesh):void 0;a.push({meshes:t,type:this._convertV0FlagToV1Type(o?.firstPersonFlag)})}),new ft(t,a)})}_convertV0FlagToV1Type(e){return e===`FirstPersonOnly`?`firstPersonOnly`:e===`ThirdPersonOnly`?`thirdPersonOnly`:e===`Both`?`both`:`auto`}},ht=new i,gt=new i,_t=new s,vt=class extends b{constructor(e){super(),this.vrmHumanoid=e,this._boneAxesMap=new Map,Object.values(e.humanBones).forEach(e=>{let t=new ae(1);t.matrixAutoUpdate=!1,t.material.depthTest=!1,t.material.depthWrite=!1,this.add(t),this._boneAxesMap.set(e,t)})}dispose(){Array.from(this._boneAxesMap.values()).forEach(e=>{e.geometry.dispose(),e.material.dispose()})}updateMatrixWorld(e){Array.from(this._boneAxesMap.entries()).forEach(([e,t])=>{e.node.updateWorldMatrix(!0,!1),e.node.matrixWorld.decompose(ht,_t,gt);let n=ht.set(.1,.1,.1).divide(gt);t.matrix.copy(e.node.matrixWorld).scale(n)}),super.updateMatrixWorld(e)}},yt=`hips.spine.chest.upperChest.neck.head.leftEye.rightEye.jaw.leftUpperLeg.leftLowerLeg.leftFoot.leftToes.rightUpperLeg.rightLowerLeg.rightFoot.rightToes.leftShoulder.leftUpperArm.leftLowerArm.leftHand.rightShoulder.rightUpperArm.rightLowerArm.rightHand.leftThumbMetacarpal.leftThumbProximal.leftThumbDistal.leftIndexProximal.leftIndexIntermediate.leftIndexDistal.leftMiddleProximal.leftMiddleIntermediate.leftMiddleDistal.leftRingProximal.leftRingIntermediate.leftRingDistal.leftLittleProximal.leftLittleIntermediate.leftLittleDistal.rightThumbMetacarpal.rightThumbProximal.rightThumbDistal.rightIndexProximal.rightIndexIntermediate.rightIndexDistal.rightMiddleProximal.rightMiddleIntermediate.rightMiddleDistal.rightRingProximal.rightRingIntermediate.rightRingDistal.rightLittleProximal.rightLittleIntermediate.rightLittleDistal`.split(`.`),bt={hips:null,spine:`hips`,chest:`spine`,upperChest:`chest`,neck:`upperChest`,head:`neck`,leftEye:`head`,rightEye:`head`,jaw:`head`,leftUpperLeg:`hips`,leftLowerLeg:`leftUpperLeg`,leftFoot:`leftLowerLeg`,leftToes:`leftFoot`,rightUpperLeg:`hips`,rightLowerLeg:`rightUpperLeg`,rightFoot:`rightLowerLeg`,rightToes:`rightFoot`,leftShoulder:`upperChest`,leftUpperArm:`leftShoulder`,leftLowerArm:`leftUpperArm`,leftHand:`leftLowerArm`,rightShoulder:`upperChest`,rightUpperArm:`rightShoulder`,rightLowerArm:`rightUpperArm`,rightHand:`rightLowerArm`,leftThumbMetacarpal:`leftHand`,leftThumbProximal:`leftThumbMetacarpal`,leftThumbDistal:`leftThumbProximal`,leftIndexProximal:`leftHand`,leftIndexIntermediate:`leftIndexProximal`,leftIndexDistal:`leftIndexIntermediate`,leftMiddleProximal:`leftHand`,leftMiddleIntermediate:`leftMiddleProximal`,leftMiddleDistal:`leftMiddleIntermediate`,leftRingProximal:`leftHand`,leftRingIntermediate:`leftRingProximal`,leftRingDistal:`leftRingIntermediate`,leftLittleProximal:`leftHand`,leftLittleIntermediate:`leftLittleProximal`,leftLittleDistal:`leftLittleIntermediate`,rightThumbMetacarpal:`rightHand`,rightThumbProximal:`rightThumbMetacarpal`,rightThumbDistal:`rightThumbProximal`,rightIndexProximal:`rightHand`,rightIndexIntermediate:`rightIndexProximal`,rightIndexDistal:`rightIndexIntermediate`,rightMiddleProximal:`rightHand`,rightMiddleIntermediate:`rightMiddleProximal`,rightMiddleDistal:`rightMiddleIntermediate`,rightRingProximal:`rightHand`,rightRingIntermediate:`rightRingProximal`,rightRingDistal:`rightRingIntermediate`,rightLittleProximal:`rightHand`,rightLittleIntermediate:`rightLittleProximal`,rightLittleDistal:`rightLittleIntermediate`};function xt(e){return e.invert?e.invert():e.inverse(),e}var I=new i,L=new s,St=class{constructor(e){this.humanBones=e,this.restPose=this.getAbsolutePose()}getAbsolutePose(){let e={};return Object.keys(this.humanBones).forEach(t=>{let n=t,r=this.getBoneNode(n);r&&(I.copy(r.position),L.copy(r.quaternion),e[n]={position:I.toArray(),rotation:L.toArray()})}),e}getPose(){let e={};return Object.keys(this.humanBones).forEach(t=>{let n=t,r=this.getBoneNode(n);if(!r)return;I.set(0,0,0),L.identity();let i=this.restPose[n];i?.position&&I.fromArray(i.position).negate(),i?.rotation&&xt(L.fromArray(i.rotation)),I.add(r.position),L.premultiply(r.quaternion),e[n]={position:I.toArray(),rotation:L.toArray()}}),e}setPose(e){Object.entries(e).forEach(([e,t])=>{let n=e,r=this.getBoneNode(n);if(!r)return;let i=this.restPose[n];i&&(t?.position&&(r.position.fromArray(t.position),i.position&&r.position.add(I.fromArray(i.position))),t?.rotation&&(r.quaternion.fromArray(t.rotation),i.rotation&&r.quaternion.multiply(L.fromArray(i.rotation))))})}resetPose(){Object.entries(this.restPose).forEach(([e,t])=>{let n=this.getBoneNode(e);n&&(t?.position&&n.position.fromArray(t.position),t?.rotation&&n.quaternion.fromArray(t.rotation))})}getBone(e){return this.humanBones[e]??void 0}getBoneNode(e){return this.humanBones[e]?.node??null}},Ct=new i,wt=new s,Tt=new i,Et=class e extends St{static _setupTransforms(e){let t=new n;t.name=`VRMHumanoidRig`;let r={},a={},o={},c={};yt.forEach(t=>{var n;let l=e.getBoneNode(t);if(l){let e=new i,u=new s;l.updateWorldMatrix(!0,!1),l.matrixWorld.decompose(e,u,Ct),r[t]=e,a[t]=u,o[t]=l.quaternion.clone();let d=new s;(n=l.parent)==null||n.matrixWorld.decompose(Ct,d,Ct),c[t]=d}});let l={};return yt.forEach(i=>{let a=e.getBoneNode(i);if(a){let e=r[i],o=i,s;for(;s==null&&(o=bt[o],o!=null);)s=r[o];let c=new n;c.name=`Normalized_`+a.name,(o?l[o]?.node:t).add(c),c.position.copy(e),s&&c.position.sub(s),l[i]={node:c}}}),{rigBones:l,root:t,parentWorldRotations:c,boneRotations:o}}constructor(t){let{rigBones:n,root:r,parentWorldRotations:i,boneRotations:a}=e._setupTransforms(t);super(n),this.original=t,this.root=r,this._parentWorldRotations=i,this._boneRotations=a}update(){yt.forEach(e=>{let t=this.original.getBoneNode(e);if(t!=null){let n=this.getBoneNode(e),r=this._parentWorldRotations[e],i=wt.copy(r).invert(),a=this._boneRotations[e];if(t.quaternion.copy(n.quaternion).multiply(r).premultiply(i).multiply(a),e===`hips`){let e=n.getWorldPosition(Tt);t.parent.updateWorldMatrix(!0,!1);let r=t.parent.matrixWorld,i=e.applyMatrix4(r.invert());t.position.copy(i)}}})}},Dt=class e{get restPose(){return this.rawRestPose}get rawRestPose(){return this._rawHumanBones.restPose}get normalizedRestPose(){return this._normalizedHumanBones.restPose}get humanBones(){return this._rawHumanBones.humanBones}get rawHumanBones(){return this._rawHumanBones.humanBones}get normalizedHumanBones(){return this._normalizedHumanBones.humanBones}get normalizedHumanBonesRoot(){return this._normalizedHumanBones.root}constructor(e,t){this.autoUpdateHumanBones=t?.autoUpdateHumanBones??!0,this._rawHumanBones=new St(e),this._normalizedHumanBones=new Et(this._rawHumanBones)}copy(e){return this.autoUpdateHumanBones=e.autoUpdateHumanBones,this._rawHumanBones=new St(e.humanBones),this._normalizedHumanBones=new Et(this._rawHumanBones),this}clone(){return new e(this.humanBones,{autoUpdateHumanBones:this.autoUpdateHumanBones}).copy(this)}getAbsolutePose(){return this.getRawAbsolutePose()}getRawAbsolutePose(){return this._rawHumanBones.getAbsolutePose()}getNormalizedAbsolutePose(){return this._normalizedHumanBones.getAbsolutePose()}getPose(){return this.getRawPose()}getRawPose(){return this._rawHumanBones.getPose()}getNormalizedPose(){return this._normalizedHumanBones.getPose()}setPose(e){return this.setRawPose(e)}setRawPose(e){return this._rawHumanBones.setPose(e)}setNormalizedPose(e){return this._normalizedHumanBones.setPose(e)}resetPose(){return this.resetRawPose()}resetRawPose(){return this._rawHumanBones.resetPose()}resetNormalizedPose(){return this._normalizedHumanBones.resetPose()}getBone(e){return this.getRawBone(e)}getRawBone(e){return this._rawHumanBones.getBone(e)}getNormalizedBone(e){return this._normalizedHumanBones.getBone(e)}getBoneNode(e){return this.getRawBoneNode(e)}getRawBoneNode(e){return this._rawHumanBones.getBoneNode(e)}getNormalizedBoneNode(e){return this._normalizedHumanBones.getBoneNode(e)}update(){this.autoUpdateHumanBones&&this._normalizedHumanBones.update()}},Ot={Hips:`hips`,Spine:`spine`,Head:`head`,LeftUpperLeg:`leftUpperLeg`,LeftLowerLeg:`leftLowerLeg`,LeftFoot:`leftFoot`,RightUpperLeg:`rightUpperLeg`,RightLowerLeg:`rightLowerLeg`,RightFoot:`rightFoot`,LeftUpperArm:`leftUpperArm`,LeftLowerArm:`leftLowerArm`,LeftHand:`leftHand`,RightUpperArm:`rightUpperArm`,RightLowerArm:`rightLowerArm`,RightHand:`rightHand`},kt=new Set([`1.0`,`1.0-beta`]),At={leftThumbProximal:`leftThumbMetacarpal`,leftThumbIntermediate:`leftThumbProximal`,rightThumbProximal:`rightThumbMetacarpal`,rightThumbIntermediate:`rightThumbProximal`},jt=class{get name(){return`VRMHumanoidLoaderPlugin`}constructor(e,t){this.parser=e,this.helperRoot=t?.helperRoot,this.autoUpdateHumanBones=t?.autoUpdateHumanBones}afterRoot(e){return F(this,null,function*(){e.userData.vrmHumanoid=yield this._import(e)})}_import(e){return F(this,null,function*(){return(yield this._v1Import(e))||(yield this._v0Import(e))||null})}_v1Import(e){return F(this,null,function*(){let t=this.parser.json;if(t.extensionsUsed?.indexOf(`VRMC_vrm`)===-1)return null;let n=t.extensions?.VRMC_vrm;if(!n)return null;let r=n.specVersion;if(!kt.has(r))return null;let i=n.humanoid;if(!i)return null;let a=i.humanBones.leftThumbIntermediate!=null||i.humanBones.rightThumbIntermediate!=null,o={};i.humanBones!=null&&(yield Promise.all(Object.entries(i.humanBones).map(e=>F(this,[e],function*([e,t]){let n=e,r=t.node;if(a){let e=At[n];e!=null&&(n=e)}let i=yield this.parser.getDependency(`node`,r);i!=null&&(o[n]={node:i})}))));let s=new Dt(this._ensureRequiredBonesExist(o),{autoUpdateHumanBones:this.autoUpdateHumanBones});if(e.scene.add(s.normalizedHumanBonesRoot),this.helperRoot){let e=new vt(s);this.helperRoot.add(e),e.renderOrder=this.helperRoot.renderOrder}return s})}_v0Import(e){return F(this,null,function*(){let t=this.parser.json.extensions?.VRM;if(!t)return null;let n=t.humanoid;if(!n)return null;let r={};n.humanBones!=null&&(yield Promise.all(n.humanBones.map(e=>F(this,null,function*(){let t=e.bone,n=e.node;if(t==null||n==null||n<0)return;let i=yield this.parser.getDependency(`node`,n);if(i==null)return;let a=At[t]??t;r[a]??(r[a]={node:i})}))));let i=new Dt(this._ensureRequiredBonesExist(r),{autoUpdateHumanBones:this.autoUpdateHumanBones});if(e.scene.add(i.normalizedHumanBonesRoot),this.helperRoot){let e=new vt(i);this.helperRoot.add(e),e.renderOrder=this.helperRoot.renderOrder}return i})}_ensureRequiredBonesExist(e){let t=Object.values(Ot).filter(t=>e[t]==null);if(t.length>0)throw Error(`VRMHumanoidLoaderPlugin: These humanoid bones are required but not exist: ${t.join(`, `)}`);return e}},Mt=class extends y{constructor(){super(),this._currentTheta=0,this._currentRadius=0,this.theta=0,this.radius=0,this._currentTheta=0,this._currentRadius=0,this._attrPos=new m(new Float32Array(195),3),this.setAttribute(`position`,this._attrPos),this._attrIndex=new m(new Uint16Array(189),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;this._currentTheta!==this.theta&&(this._currentTheta=this.theta,e=!0),this._currentRadius!==this.radius&&(this._currentRadius=this.radius,e=!0),e&&this._buildPosition()}_buildPosition(){this._attrPos.setXYZ(0,0,0,0);for(let e=0;e<64;e++){let t=e/63*this._currentTheta;this._attrPos.setXYZ(e+1,this._currentRadius*Math.sin(t),0,this._currentRadius*Math.cos(t))}this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<63;e++)this._attrIndex.setXYZ(e*3,0,e+1,e+2);this._attrIndex.needsUpdate=!0}},Nt=class extends y{constructor(){super(),this.radius=0,this._currentRadius=0,this.tail=new i,this._currentTail=new i,this._attrPos=new m(new Float32Array(294),3),this.setAttribute(`position`,this._attrPos),this._attrIndex=new m(new Uint16Array(194),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;this._currentRadius!==this.radius&&(this._currentRadius=this.radius,e=!0),this._currentTail.equals(this.tail)||(this._currentTail.copy(this.tail),e=!0),e&&this._buildPosition()}_buildPosition(){for(let e=0;e<32;e++){let t=e/16*Math.PI;this._attrPos.setXYZ(e,Math.cos(t),Math.sin(t),0),this._attrPos.setXYZ(32+e,0,Math.cos(t),Math.sin(t)),this._attrPos.setXYZ(64+e,Math.sin(t),0,Math.cos(t))}this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.setXYZ(96,0,0,0),this._attrPos.setXYZ(97,this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<32;e++){let t=(e+1)%32;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(64+e*2,32+e,32+t),this._attrIndex.setXY(128+e*2,64+e,64+t)}this._attrIndex.setXY(192,96,97),this._attrIndex.needsUpdate=!0}},Pt=new s,Ft=new s,R=new i,It=new i,Lt=Math.sqrt(2)/2,Rt=new s(0,0,-Lt,Lt),zt=new i(0,1,0),Bt=class extends b{constructor(e){super(),this.matrixAutoUpdate=!1,this.vrmLookAt=e;{let e=new Mt;e.radius=.5;let t=new r({color:65280,transparent:!0,opacity:.5,side:2,depthTest:!1,depthWrite:!1});this._meshPitch=new u(e,t),this.add(this._meshPitch)}{let e=new Mt;e.radius=.5;let t=new r({color:16711680,transparent:!0,opacity:.5,side:2,depthTest:!1,depthWrite:!1});this._meshYaw=new u(e,t),this.add(this._meshYaw)}{let e=new Nt;e.radius=.1;let t=new ce({color:16777215,depthTest:!1,depthWrite:!1});this._lineTarget=new p(e,t),this._lineTarget.frustumCulled=!1,this.add(this._lineTarget)}}dispose(){this._meshYaw.geometry.dispose(),this._meshYaw.material.dispose(),this._meshPitch.geometry.dispose(),this._meshPitch.material.dispose(),this._lineTarget.geometry.dispose(),this._lineTarget.material.dispose()}updateMatrixWorld(e){let t=l.DEG2RAD*this.vrmLookAt.yaw;this._meshYaw.geometry.theta=t,this._meshYaw.geometry.update();let n=l.DEG2RAD*this.vrmLookAt.pitch;this._meshPitch.geometry.theta=n,this._meshPitch.geometry.update(),this.vrmLookAt.getLookAtWorldPosition(R),this.vrmLookAt.getLookAtWorldQuaternion(Pt),Pt.multiply(this.vrmLookAt.getFaceFrontQuaternion(Ft)),this._meshYaw.position.copy(R),this._meshYaw.quaternion.copy(Pt),this._meshPitch.position.copy(R),this._meshPitch.quaternion.copy(Pt),this._meshPitch.quaternion.multiply(Ft.setFromAxisAngle(zt,t)),this._meshPitch.quaternion.multiply(Rt);let{target:r,autoUpdate:i}=this.vrmLookAt;r!=null&&i&&(r.getWorldPosition(It).sub(R),this._lineTarget.geometry.tail.copy(It),this._lineTarget.geometry.update(),this._lineTarget.position.copy(R)),super.updateMatrixWorld(e)}},Vt=new i,Ht=new i;function Ut(e,t){return e.matrixWorld.decompose(Vt,t,Ht),t}function Wt(e){return[Math.atan2(-e.z,e.x),Math.atan2(e.y,Math.sqrt(e.x*e.x+e.z*e.z))]}function Gt(e){let t=Math.round(e/2/Math.PI);return e-2*Math.PI*t}var Kt=new i(0,0,1),qt=new i,Jt=new i,Yt=new i,Xt=new s,Zt=new s,Qt=new s,$t=new s,en=new g,tn=class e{constructor(e,t){this.offsetFromHeadBone=new i,this.autoUpdate=!0,this.faceFront=new i(0,0,1),this.humanoid=e,this.applier=t,this._yaw=0,this._pitch=0,this._needsUpdate=!0,this._restHeadWorldQuaternion=this.getLookAtWorldQuaternion(new s)}get yaw(){return this._yaw}set yaw(e){this._yaw=e,this._needsUpdate=!0}get pitch(){return this._pitch}set pitch(e){this._pitch=e,this._needsUpdate=!0}get euler(){return this.getEuler(new g)}getEuler(e){return e.set(l.DEG2RAD*this._pitch,l.DEG2RAD*this._yaw,0,`YXZ`)}copy(e){if(this.humanoid!==e.humanoid)throw Error(`VRMLookAt: humanoid must be same in order to copy`);return this.offsetFromHeadBone.copy(e.offsetFromHeadBone),this.applier=e.applier,this.autoUpdate=e.autoUpdate,this.target=e.target,this.faceFront.copy(e.faceFront),this}clone(){return new e(this.humanoid,this.applier).copy(this)}reset(){this._yaw=0,this._pitch=0,this._needsUpdate=!0}getLookAtWorldPosition(e){let t=this.humanoid.getRawBoneNode(`head`);return e.copy(this.offsetFromHeadBone).applyMatrix4(t.matrixWorld)}getLookAtWorldQuaternion(e){return Ut(this.humanoid.getRawBoneNode(`head`),e)}getFaceFrontQuaternion(e){if(this.faceFront.distanceToSquared(Kt)<.01)return e.copy(this._restHeadWorldQuaternion).invert();let[t,n]=Wt(this.faceFront);return en.set(0,.5*Math.PI+t,n,`YZX`),e.setFromEuler(en).premultiply($t.copy(this._restHeadWorldQuaternion).invert())}getLookAtWorldDirection(e){return this.getLookAtWorldQuaternion(Zt),this.getFaceFrontQuaternion(Qt),e.copy(Kt).applyQuaternion(Zt).applyQuaternion(Qt).applyEuler(this.getEuler(en))}lookAt(e){let t=Xt.copy(this._restHeadWorldQuaternion).multiply(xt(this.getLookAtWorldQuaternion(Zt))),n=this.getLookAtWorldPosition(Jt),r=Yt.copy(e).sub(n).applyQuaternion(t).normalize(),[i,a]=Wt(this.faceFront),[o,s]=Wt(r),c=Gt(o-i),u=Gt(a-s);this._yaw=l.RAD2DEG*c,this._pitch=l.RAD2DEG*u,this._needsUpdate=!0}update(e){this.target!=null&&this.autoUpdate&&this.lookAt(this.target.getWorldPosition(qt)),this._needsUpdate&&(this._needsUpdate=!1,this.applier.applyYawPitch(this._yaw,this._pitch))}};tn.EULER_ORDER=`YXZ`;var nn=tn,rn=new i(0,0,1),z=new s,B=new s,V=new g(0,0,0,`YXZ`),an=class{constructor(e,t,n,r,a){this.humanoid=e,this.rangeMapHorizontalInner=t,this.rangeMapHorizontalOuter=n,this.rangeMapVerticalDown=r,this.rangeMapVerticalUp=a,this.faceFront=new i(0,0,1),this._restQuatLeftEye=new s,this._restQuatRightEye=new s,this._restLeftEyeParentWorldQuat=new s,this._restRightEyeParentWorldQuat=new s;let o=this.humanoid.getRawBoneNode(`leftEye`),c=this.humanoid.getRawBoneNode(`rightEye`);o&&(this._restQuatLeftEye.copy(o.quaternion),Ut(o.parent,this._restLeftEyeParentWorldQuat)),c&&(this._restQuatRightEye.copy(c.quaternion),Ut(c.parent,this._restRightEyeParentWorldQuat))}applyYawPitch(e,t){let n=this.humanoid.getRawBoneNode(`leftEye`),r=this.humanoid.getRawBoneNode(`rightEye`),i=this.humanoid.getNormalizedBoneNode(`leftEye`),a=this.humanoid.getNormalizedBoneNode(`rightEye`);n&&(V.x=t<0?-l.DEG2RAD*this.rangeMapVerticalDown.map(-t):l.DEG2RAD*this.rangeMapVerticalUp.map(t),V.y=e<0?-l.DEG2RAD*this.rangeMapHorizontalInner.map(-e):l.DEG2RAD*this.rangeMapHorizontalOuter.map(e),z.setFromEuler(V),this._getWorldFaceFrontQuat(B),i.quaternion.copy(B).multiply(z).multiply(B.invert()),z.copy(this._restLeftEyeParentWorldQuat),n.quaternion.copy(i.quaternion).multiply(z).premultiply(z.invert()).multiply(this._restQuatLeftEye)),r&&(V.x=t<0?-l.DEG2RAD*this.rangeMapVerticalDown.map(-t):l.DEG2RAD*this.rangeMapVerticalUp.map(t),V.y=e<0?-l.DEG2RAD*this.rangeMapHorizontalOuter.map(-e):l.DEG2RAD*this.rangeMapHorizontalInner.map(e),z.setFromEuler(V),this._getWorldFaceFrontQuat(B),a.quaternion.copy(B).multiply(z).multiply(B.invert()),z.copy(this._restRightEyeParentWorldQuat),r.quaternion.copy(a.quaternion).multiply(z).premultiply(z.invert()).multiply(this._restQuatRightEye))}lookAt(e){let t=l.RAD2DEG*e.y,n=l.RAD2DEG*e.x;this.applyYawPitch(t,n)}_getWorldFaceFrontQuat(e){if(this.faceFront.distanceToSquared(rn)<.01)return e.identity();let[t,n]=Wt(this.faceFront);return V.set(0,.5*Math.PI+t,n,`YZX`),e.setFromEuler(V)}};an.type=`bone`;var on=class{constructor(e,t,n,r,i){this.expressions=e,this.rangeMapHorizontalInner=t,this.rangeMapHorizontalOuter=n,this.rangeMapVerticalDown=r,this.rangeMapVerticalUp=i}applyYawPitch(e,t){t<0?(this.expressions.setValue(`lookDown`,0),this.expressions.setValue(`lookUp`,this.rangeMapVerticalUp.map(-t))):(this.expressions.setValue(`lookUp`,0),this.expressions.setValue(`lookDown`,this.rangeMapVerticalDown.map(t))),e<0?(this.expressions.setValue(`lookLeft`,0),this.expressions.setValue(`lookRight`,this.rangeMapHorizontalOuter.map(-e))):(this.expressions.setValue(`lookRight`,0),this.expressions.setValue(`lookLeft`,this.rangeMapHorizontalOuter.map(e)))}lookAt(e){let t=l.RAD2DEG*e.y,n=l.RAD2DEG*e.x;this.applyYawPitch(t,n)}};on.type=`expression`;var sn=class{constructor(e,t){this.inputMaxValue=e,this.outputScale=t}map(e){return this.outputScale*Ze(e/this.inputMaxValue)}},cn=new Set([`1.0`,`1.0-beta`]),ln=.01,un=class{get name(){return`VRMLookAtLoaderPlugin`}constructor(e,t){this.parser=e,this.helperRoot=t?.helperRoot}afterRoot(e){return F(this,null,function*(){let t=e.userData.vrmHumanoid;if(t===null)return;if(t===void 0)throw Error(`VRMLookAtLoaderPlugin: vrmHumanoid is undefined. VRMHumanoidLoaderPlugin have to be used first`);let n=e.userData.vrmExpressionManager;if(n!==null){if(n===void 0)throw Error(`VRMLookAtLoaderPlugin: vrmExpressionManager is undefined. VRMExpressionLoaderPlugin have to be used first`);e.userData.vrmLookAt=yield this._import(e,t,n)}})}_import(e,t,n){return F(this,null,function*(){return t==null||n==null?null:(yield this._v1Import(e,t,n))||(yield this._v0Import(e,t,n))||null})}_v1Import(e,t,n){return F(this,null,function*(){let e=this.parser.json;if(e.extensionsUsed?.indexOf(`VRMC_vrm`)===-1)return null;let r=e.extensions?.VRMC_vrm;if(!r)return null;let i=r.specVersion;if(!cn.has(i))return null;let a=r.lookAt;if(!a)return null;let o=a.type===`expression`?1:10,s=this._v1ImportRangeMap(a.rangeMapHorizontalInner,o),c=this._v1ImportRangeMap(a.rangeMapHorizontalOuter,o),l=this._v1ImportRangeMap(a.rangeMapVerticalDown,o),u=this._v1ImportRangeMap(a.rangeMapVerticalUp,o),d;d=a.type===`expression`?new on(n,s,c,l,u):new an(t,s,c,l,u);let f=this._importLookAt(t,d);return f.offsetFromHeadBone.fromArray(a.offsetFromHeadBone??[0,.06,0]),f})}_v1ImportRangeMap(e,t){let n=e?.inputMaxValue??90,r=e?.outputScale??t;return n<ln&&(n=ln),new sn(n,r)}_v0Import(e,t,n){return F(this,null,function*(){let e=this.parser.json.extensions?.VRM;if(!e)return null;let r=e.firstPerson;if(!r)return null;let i=r.lookAtTypeName===`BlendShape`?1:10,a=this._v0ImportDegreeMap(r.lookAtHorizontalInner,i),o=this._v0ImportDegreeMap(r.lookAtHorizontalOuter,i),s=this._v0ImportDegreeMap(r.lookAtVerticalDown,i),c=this._v0ImportDegreeMap(r.lookAtVerticalUp,i),l;l=r.lookAtTypeName===`BlendShape`?new on(n,a,o,s,c):new an(t,a,o,s,c);let u=this._importLookAt(t,l);return r.firstPersonBoneOffset?u.offsetFromHeadBone.set(r.firstPersonBoneOffset.x??0,r.firstPersonBoneOffset.y??.06,-(r.firstPersonBoneOffset.z??0)):u.offsetFromHeadBone.set(0,.06,0),u.faceFront.set(0,0,-1),l instanceof an&&l.faceFront.set(0,0,-1),u})}_v0ImportDegreeMap(e,t){let n=e?.curve;JSON.stringify(n);let r=e?.xRange??90,i=e?.yRange??t;return r<ln&&(r=ln),new sn(r,i)}_importLookAt(e,t){let n=new nn(e,t);if(this.helperRoot){let e=new Bt(n);this.helperRoot.add(e),e.renderOrder=this.helperRoot.renderOrder}return n}};function dn(e,t){return typeof e!=`string`||e===``?``:(/^https?:\/\//i.test(t)&&/^\//.test(e)&&(t=t.replace(/(^https?:\/\/[^/]+).*/i,`$1`)),/^(https?:)?\/\//i.test(e)||/^data:.*,.*$/i.test(e)||/^blob:.*$/i.test(e)?e:t+e)}var fn=new Set([`1.0`,`1.0-beta`]),pn=class{get name(){return`VRMMetaLoaderPlugin`}constructor(e,t){this.parser=e,this.needThumbnailImage=t?.needThumbnailImage??!1,this.acceptLicenseUrls=t?.acceptLicenseUrls??[`https://vrm.dev/licenses/1.0/`],this.acceptV0Meta=t?.acceptV0Meta??!0}afterRoot(e){return F(this,null,function*(){e.userData.vrmMeta=yield this._import(e)})}_import(e){return F(this,null,function*(){return(yield this._v1Import(e))??(yield this._v0Import(e))??null})}_v1Import(e){return F(this,null,function*(){let e=this.parser.json;if(e.extensionsUsed?.indexOf(`VRMC_vrm`)===-1)return null;let t=e.extensions?.VRMC_vrm;if(t==null)return null;let n=t.specVersion;if(!fn.has(n))return null;let r=t.meta;if(!r)return null;let i=r.licenseUrl;if(!new Set(this.acceptLicenseUrls).has(i))throw Error(`VRMMetaLoaderPlugin: The license url "${i}" is not accepted`);let a;return this.needThumbnailImage&&r.thumbnailImage!=null&&(a=(yield this._extractGLTFImage(r.thumbnailImage))??void 0),{metaVersion:`1`,name:r.name,version:r.version,authors:r.authors,copyrightInformation:r.copyrightInformation,contactInformation:r.contactInformation,references:r.references,thirdPartyLicenses:r.thirdPartyLicenses,thumbnailImage:a,licenseUrl:r.licenseUrl,avatarPermission:r.avatarPermission,allowExcessivelyViolentUsage:r.allowExcessivelyViolentUsage,allowExcessivelySexualUsage:r.allowExcessivelySexualUsage,commercialUsage:r.commercialUsage,allowPoliticalOrReligiousUsage:r.allowPoliticalOrReligiousUsage,allowAntisocialOrHateUsage:r.allowAntisocialOrHateUsage,creditNotation:r.creditNotation,allowRedistribution:r.allowRedistribution,modification:r.modification,otherLicenseUrl:r.otherLicenseUrl}})}_v0Import(e){return F(this,null,function*(){let e=this.parser.json.extensions?.VRM;if(!e)return null;let t=e.meta;if(!t)return null;if(!this.acceptV0Meta)throw Error(`VRMMetaLoaderPlugin: Attempted to load VRM0.0 meta but acceptV0Meta is false`);let n;return this.needThumbnailImage&&t.texture!=null&&t.texture!==-1&&(n=yield this.parser.getDependency(`texture`,t.texture)),{metaVersion:`0`,allowedUserName:t.allowedUserName,author:t.author,commercialUssageName:t.commercialUssageName,contactInformation:t.contactInformation,licenseName:t.licenseName,otherLicenseUrl:t.otherLicenseUrl,otherPermissionUrl:t.otherPermissionUrl,reference:t.reference,sexualUssageName:t.sexualUssageName,texture:n??void 0,title:t.title,version:t.version,violentUssageName:t.violentUssageName}})}_extractGLTFImage(e){return F(this,null,function*(){let t=this.parser.json.images?.[e];if(t==null)return null;let n=t.uri;if(t.bufferView!=null){let e=yield this.parser.getDependency(`bufferView`,t.bufferView),r=new Blob([e],{type:t.mimeType});n=URL.createObjectURL(r)}return n==null?null:yield new v().loadAsync(dn(n,this.parser.options.path)).catch(e=>null)})}},mn=class{constructor(e){this.scene=e.scene,this.meta=e.meta,this.humanoid=e.humanoid,this.expressionManager=e.expressionManager,this.firstPerson=e.firstPerson,this.lookAt=e.lookAt}update(e){this.humanoid.update(),this.lookAt&&this.lookAt.update(e),this.expressionManager&&this.expressionManager.update()}},hn=class extends mn{constructor(e){super(e),this.materials=e.materials,this.springBoneManager=e.springBoneManager,this.nodeConstraintManager=e.nodeConstraintManager}update(e){super.update(e),this.nodeConstraintManager&&this.nodeConstraintManager.update(),this.springBoneManager&&this.springBoneManager.update(e),this.materials&&this.materials.forEach(t=>{t.update&&t.update(e)})}},gn=Object.defineProperty,_n=Object.getOwnPropertySymbols,vn=Object.prototype.hasOwnProperty,yn=Object.prototype.propertyIsEnumerable,bn=(e,t,n)=>t in e?gn(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,xn=(e,t)=>{for(var n in t||={})vn.call(t,n)&&bn(e,n,t[n]);if(_n)for(var n of _n(t))yn.call(t,n)&&bn(e,n,t[n]);return e},H=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())});function Sn(e,t){e.colorSpace=t}var Cn=class{get pending(){return Promise.all(this._pendings)}constructor(e,t){this._parser=e,this._materialParams=t,this._pendings=[]}assignPrimitive(e,t){t!=null&&(this._materialParams[e]=t)}assignColor(e,t,n){if(t!=null){let r=new S().fromArray(t);n&&r.convertSRGBToLinear(),this._materialParams[e]=r}}assignTexture(e,t,n){return H(this,null,function*(){let r=H(this,null,function*(){if(t!=null){let r=yield this._parser.assignTexture(this._materialParams,e,t);if(r==null)return;n&&Sn(r,`srgb`)}});return this._pendings.push(r),r})}assignTextureByIndex(e,t,n){return H(this,null,function*(){return this.assignTexture(e,t==null?void 0:{index:t},n)})}},wn=`// #define PHONG

varying vec3 vViewPosition;

#ifndef FLAT_SHADED
  varying vec3 vNormal;
#endif

#include <common>

// #include <uv_pars_vertex>
#ifdef MTOON_USE_UV
  varying vec2 vUv;

  // COMPAT: pre-r151 uses a common uvTransform
  #if THREE_VRM_THREE_REVISION < 151
    uniform mat3 uvTransform;
  #endif
#endif

// #include <uv2_pars_vertex>
// COMAPT: pre-r151 uses uv2 for lightMap and aoMap
#if THREE_VRM_THREE_REVISION < 151
  #if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )
    attribute vec2 uv2;
    varying vec2 vUv2;
    uniform mat3 uv2Transform;
  #endif
#endif

// #include <displacementmap_pars_vertex>
// #include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>

#ifdef USE_OUTLINEWIDTHMULTIPLYTEXTURE
  uniform sampler2D outlineWidthMultiplyTexture;
  uniform mat3 outlineWidthMultiplyTextureUvTransform;
#endif

uniform float outlineWidthFactor;

void main() {

  // #include <uv_vertex>
  #ifdef MTOON_USE_UV
    // COMPAT: pre-r151 uses a common uvTransform
    #if THREE_VRM_THREE_REVISION >= 151
      vUv = uv;
    #else
      vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
    #endif
  #endif

  // #include <uv2_vertex>
  // COMAPT: pre-r151 uses uv2 for lightMap and aoMap
  #if THREE_VRM_THREE_REVISION < 151
    #if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )
      vUv2 = ( uv2Transform * vec3( uv2, 1 ) ).xy;
    #endif
  #endif

  #include <color_vertex>

  #include <beginnormal_vertex>
  #include <morphnormal_vertex>
  #include <skinbase_vertex>
  #include <skinnormal_vertex>

  // we need this to compute the outline properly
  objectNormal = normalize( objectNormal );

  #include <defaultnormal_vertex>

  #ifndef FLAT_SHADED // Normal computed with derivatives when FLAT_SHADED
    vNormal = normalize( transformedNormal );
  #endif

  #include <begin_vertex>

  #include <morphtarget_vertex>
  #include <skinning_vertex>
  // #include <displacementmap_vertex>
  #include <project_vertex>
  #include <logdepthbuf_vertex>
  #include <clipping_planes_vertex>

  vViewPosition = - mvPosition.xyz;

  #ifdef OUTLINE
    float worldNormalLength = length( transformedNormal );
    vec3 outlineOffset = outlineWidthFactor * worldNormalLength * objectNormal;

    #ifdef USE_OUTLINEWIDTHMULTIPLYTEXTURE
      vec2 outlineWidthMultiplyTextureUv = ( outlineWidthMultiplyTextureUvTransform * vec3( vUv, 1 ) ).xy;
      float outlineTex = texture2D( outlineWidthMultiplyTexture, outlineWidthMultiplyTextureUv ).g;
      outlineOffset *= outlineTex;
    #endif

    #ifdef OUTLINE_WIDTH_SCREEN
      outlineOffset *= vViewPosition.z / projectionMatrix[ 1 ].y;
    #endif

    gl_Position = projectionMatrix * modelViewMatrix * vec4( outlineOffset + transformed, 1.0 );

    gl_Position.z += 1E-6 * gl_Position.w; // anti-artifact magic
  #endif

  #include <worldpos_vertex>
  // #include <envmap_vertex>
  #include <shadowmap_vertex>
  #include <fog_vertex>

}`,Tn=`// #define PHONG

uniform vec3 litFactor;

uniform float opacity;

uniform vec3 shadeColorFactor;
#ifdef USE_SHADEMULTIPLYTEXTURE
  uniform sampler2D shadeMultiplyTexture;
  uniform mat3 shadeMultiplyTextureUvTransform;
#endif

uniform float shadingShiftFactor;
uniform float shadingToonyFactor;

#ifdef USE_SHADINGSHIFTTEXTURE
  uniform sampler2D shadingShiftTexture;
  uniform mat3 shadingShiftTextureUvTransform;
  uniform float shadingShiftTextureScale;
#endif

uniform float giEqualizationFactor;

uniform vec3 parametricRimColorFactor;
#ifdef USE_RIMMULTIPLYTEXTURE
  uniform sampler2D rimMultiplyTexture;
  uniform mat3 rimMultiplyTextureUvTransform;
#endif
uniform float rimLightingMixFactor;
uniform float parametricRimFresnelPowerFactor;
uniform float parametricRimLiftFactor;

#ifdef USE_MATCAPTEXTURE
  uniform vec3 matcapFactor;
  uniform sampler2D matcapTexture;
  uniform mat3 matcapTextureUvTransform;
#endif

uniform vec3 emissive;
uniform float emissiveIntensity;

uniform vec3 outlineColorFactor;
uniform float outlineLightingMixFactor;

#ifdef USE_UVANIMATIONMASKTEXTURE
  uniform sampler2D uvAnimationMaskTexture;
  uniform mat3 uvAnimationMaskTextureUvTransform;
#endif

uniform float uvAnimationScrollXOffset;
uniform float uvAnimationScrollYOffset;
uniform float uvAnimationRotationPhase;

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>

// #include <uv_pars_fragment>
#if ( defined( MTOON_USE_UV ) && !defined( MTOON_UVS_VERTEX_ONLY ) )
  varying vec2 vUv;
#endif

// #include <uv2_pars_fragment>
// COMAPT: pre-r151 uses uv2 for lightMap and aoMap
#if THREE_VRM_THREE_REVISION < 151
  #if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )
    varying vec2 vUv2;
  #endif
#endif

#include <map_pars_fragment>

#ifdef USE_MAP
  uniform mat3 mapUvTransform;
#endif

// #include <alphamap_pars_fragment>

#include <alphatest_pars_fragment>

#include <aomap_pars_fragment>
// #include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>

#ifdef USE_EMISSIVEMAP
  uniform mat3 emissiveMapUvTransform;
#endif

// #include <envmap_common_pars_fragment>
// #include <envmap_pars_fragment>
// #include <cube_uv_reflection_fragment>
#include <fog_pars_fragment>

// #include <bsdfs>
// COMPAT: pre-r151 doesn't have BRDF_Lambert in <common>
#if THREE_VRM_THREE_REVISION < 151
  vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
    return RECIPROCAL_PI * diffuseColor;
  }
#endif

#include <lights_pars_begin>

#include <normal_pars_fragment>

// #include <lights_phong_pars_fragment>
varying vec3 vViewPosition;

struct MToonMaterial {
  vec3 diffuseColor;
  vec3 shadeColor;
  float shadingShift;
};

float linearstep( float a, float b, float t ) {
  return clamp( ( t - a ) / ( b - a ), 0.0, 1.0 );
}

/**
 * Convert NdotL into toon shading factor using shadingShift and shadingToony
 */
float getShading(
  const in float dotNL,
  const in float shadow,
  const in float shadingShift
) {
  float shading = dotNL;
  shading = shading + shadingShift;
  shading = linearstep( -1.0 + shadingToonyFactor, 1.0 - shadingToonyFactor, shading );
  shading *= shadow;
  return shading;
}

/**
 * Mix diffuseColor and shadeColor using shading factor and light color
 */
vec3 getDiffuse(
  const in MToonMaterial material,
  const in float shading,
  in vec3 lightColor
) {
  #ifdef DEBUG_LITSHADERATE
    return vec3( BRDF_Lambert( shading * lightColor ) );
  #endif

  vec3 col = lightColor * BRDF_Lambert( mix( material.shadeColor, material.diffuseColor, shading ) );

  // The "comment out if you want to PBR absolutely" line
  #ifdef V0_COMPAT_SHADE
    col = min( col, material.diffuseColor );
  #endif

  return col;
}

// COMPAT: pre-r156 uses a struct GeometricContext
#if THREE_VRM_THREE_REVISION >= 157
  void RE_Direct_MToon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in MToonMaterial material, const in float shadow, inout ReflectedLight reflectedLight ) {
    float dotNL = clamp( dot( geometryNormal, directLight.direction ), -1.0, 1.0 );
    vec3 irradiance = directLight.color;

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;

    irradiance *= dotNL;

    float shading = getShading( dotNL, shadow, material.shadingShift );

    // toon shaded diffuse
    reflectedLight.directDiffuse += getDiffuse( material, shading, directLight.color );
  }

  void RE_IndirectDiffuse_MToon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in MToonMaterial material, inout ReflectedLight reflectedLight ) {
    // indirect diffuse will use diffuseColor, no shadeColor involved
    reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;
  }
#else
  void RE_Direct_MToon( const in IncidentLight directLight, const in GeometricContext geometry, const in MToonMaterial material, const in float shadow, inout ReflectedLight reflectedLight ) {
    float dotNL = clamp( dot( geometry.normal, directLight.direction ), -1.0, 1.0 );
    vec3 irradiance = directLight.color;

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;

    irradiance *= dotNL;

    float shading = getShading( dotNL, shadow, material.shadingShift );

    // toon shaded diffuse
    reflectedLight.directDiffuse += getDiffuse( material, shading, directLight.color );
  }

  void RE_IndirectDiffuse_MToon( const in vec3 irradiance, const in GeometricContext geometry, const in MToonMaterial material, inout ReflectedLight reflectedLight ) {
    // indirect diffuse will use diffuseColor, no shadeColor involved
    reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );

    // directSpecular will be used for rim lighting, not an actual specular
    reflectedLight.directSpecular += irradiance;
  }
#endif

#define RE_Direct RE_Direct_MToon
#define RE_IndirectDiffuse RE_IndirectDiffuse_MToon
#define Material_LightProbeLOD( material ) (0)

#include <shadowmap_pars_fragment>
// #include <bumpmap_pars_fragment>

// #include <normalmap_pars_fragment>
#ifdef USE_NORMALMAP

  uniform sampler2D normalMap;
  uniform mat3 normalMapUvTransform;
  uniform vec2 normalScale;

#endif

// COMPAT: pre-r151
// USE_NORMALMAP_OBJECTSPACE used to be OBJECTSPACE_NORMALMAP in pre-r151
#if defined( USE_NORMALMAP_OBJECTSPACE ) || defined( OBJECTSPACE_NORMALMAP )

  uniform mat3 normalMatrix;

#endif

// COMPAT: pre-r151
// USE_NORMALMAP_TANGENTSPACE used to be TANGENTSPACE_NORMALMAP in pre-r151
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( TANGENTSPACE_NORMALMAP ) )

  // Per-Pixel Tangent Space Normal Mapping
  // http://hacksoflife.blogspot.ch/2009/11/per-pixel-tangent-space-normal-mapping.html

  // three-vrm specific change: it requires \`uv\` as an input in order to support uv scrolls

  // Temporary compat against shader change @ Three.js r126, r151
  #if THREE_VRM_THREE_REVISION >= 151

    mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {

      vec3 q0 = dFdx( eye_pos.xyz );
      vec3 q1 = dFdy( eye_pos.xyz );
      vec2 st0 = dFdx( uv.st );
      vec2 st1 = dFdy( uv.st );

      vec3 N = surf_norm;

      vec3 q1perp = cross( q1, N );
      vec3 q0perp = cross( N, q0 );

      vec3 T = q1perp * st0.x + q0perp * st1.x;
      vec3 B = q1perp * st0.y + q0perp * st1.y;

      float det = max( dot( T, T ), dot( B, B ) );
      float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );

      return mat3( T * scale, B * scale, N );

    }

  #else

    vec3 perturbNormal2Arb( vec2 uv, vec3 eye_pos, vec3 surf_norm, vec3 mapN, float faceDirection ) {

      vec3 q0 = vec3( dFdx( eye_pos.x ), dFdx( eye_pos.y ), dFdx( eye_pos.z ) );
      vec3 q1 = vec3( dFdy( eye_pos.x ), dFdy( eye_pos.y ), dFdy( eye_pos.z ) );
      vec2 st0 = dFdx( uv.st );
      vec2 st1 = dFdy( uv.st );

      vec3 N = normalize( surf_norm );

      vec3 q1perp = cross( q1, N );
      vec3 q0perp = cross( N, q0 );

      vec3 T = q1perp * st0.x + q0perp * st1.x;
      vec3 B = q1perp * st0.y + q0perp * st1.y;

      // three-vrm specific change: Workaround for the issue that happens when delta of uv = 0.0
      // TODO: Is this still required? Or shall I make a PR about it?
      if ( length( T ) == 0.0 || length( B ) == 0.0 ) {
        return surf_norm;
      }

      float det = max( dot( T, T ), dot( B, B ) );
      float scale = ( det == 0.0 ) ? 0.0 : faceDirection * inversesqrt( det );

      return normalize( T * ( mapN.x * scale ) + B * ( mapN.y * scale ) + N * mapN.z );

    }

  #endif

#endif

// #include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

// == post correction ==========================================================
void postCorrection() {
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
  #include <premultiplied_alpha_fragment>
  #include <dithering_fragment>
}

// == main procedure ===========================================================
void main() {
  #include <clipping_planes_fragment>

  vec2 uv = vec2(0.5, 0.5);

  #if ( defined( MTOON_USE_UV ) && !defined( MTOON_UVS_VERTEX_ONLY ) )
    uv = vUv;

    float uvAnimMask = 1.0;
    #ifdef USE_UVANIMATIONMASKTEXTURE
      vec2 uvAnimationMaskTextureUv = ( uvAnimationMaskTextureUvTransform * vec3( uv, 1 ) ).xy;
      uvAnimMask = texture2D( uvAnimationMaskTexture, uvAnimationMaskTextureUv ).b;
    #endif

    float uvRotCos = cos( uvAnimationRotationPhase * uvAnimMask );
    float uvRotSin = sin( uvAnimationRotationPhase * uvAnimMask );
    uv = mat2( uvRotCos, -uvRotSin, uvRotSin, uvRotCos ) * ( uv - 0.5 ) + 0.5;
    uv = uv + vec2( uvAnimationScrollXOffset, uvAnimationScrollYOffset ) * uvAnimMask;
  #endif

  #ifdef DEBUG_UV
    gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
    #if ( defined( MTOON_USE_UV ) && !defined( MTOON_UVS_VERTEX_ONLY ) )
      gl_FragColor = vec4( uv, 0.0, 1.0 );
    #endif
    return;
  #endif

  vec4 diffuseColor = vec4( litFactor, opacity );
  ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
  vec3 totalEmissiveRadiance = emissive * emissiveIntensity;

  #include <logdepthbuf_fragment>

  // #include <map_fragment>
  #ifdef USE_MAP
    vec2 mapUv = ( mapUvTransform * vec3( uv, 1 ) ).xy;
    vec4 sampledDiffuseColor = texture2D( map, mapUv );
    #ifdef DECODE_VIDEO_TEXTURE
      sampledDiffuseColor = vec4( mix( pow( sampledDiffuseColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), sampledDiffuseColor.rgb * 0.0773993808, vec3( lessThanEqual( sampledDiffuseColor.rgb, vec3( 0.04045 ) ) ) ), sampledDiffuseColor.w );
    #endif
    diffuseColor *= sampledDiffuseColor;
  #endif

  // #include <color_fragment>
  #if ( defined( USE_COLOR ) && !defined( IGNORE_VERTEX_COLOR ) )
    diffuseColor.rgb *= vColor;
  #endif

  // #include <alphamap_fragment>

  #include <alphatest_fragment>

  // #include <specularmap_fragment>

  // #include <normal_fragment_begin>
  float faceDirection = gl_FrontFacing ? 1.0 : -1.0;

  #ifdef FLAT_SHADED

    vec3 fdx = dFdx( vViewPosition );
    vec3 fdy = dFdy( vViewPosition );
    vec3 normal = normalize( cross( fdx, fdy ) );

  #else

    vec3 normal = normalize( vNormal );

    #ifdef DOUBLE_SIDED

      normal *= faceDirection;

    #endif

  #endif

  #ifdef USE_NORMALMAP

    vec2 normalMapUv = ( normalMapUvTransform * vec3( uv, 1 ) ).xy;

  #endif

  #ifdef USE_NORMALMAP_TANGENTSPACE

    #ifdef USE_TANGENT

      mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );

    #else

      mat3 tbn = getTangentFrame( - vViewPosition, normal, normalMapUv );

    #endif

    #if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )

      tbn[0] *= faceDirection;
      tbn[1] *= faceDirection;

    #endif

  #endif

  #ifdef USE_CLEARCOAT_NORMALMAP

    #ifdef USE_TANGENT

      mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );

    #else

      mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );

    #endif

    #if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )

      tbn2[0] *= faceDirection;
      tbn2[1] *= faceDirection;

    #endif

  #endif

  // non perturbed normal for clearcoat among others

  vec3 nonPerturbedNormal = normal;

  #ifdef OUTLINE
    normal *= -1.0;
  #endif

  // #include <normal_fragment_maps>

  // COMPAT: pre-r151
  // USE_NORMALMAP_OBJECTSPACE used to be OBJECTSPACE_NORMALMAP in pre-r151
  #if defined( USE_NORMALMAP_OBJECTSPACE ) || defined( OBJECTSPACE_NORMALMAP )

    normal = texture2D( normalMap, normalMapUv ).xyz * 2.0 - 1.0; // overrides both flatShading and attribute normals

    #ifdef FLIP_SIDED

      normal = - normal;

    #endif

    #ifdef DOUBLE_SIDED

      normal = normal * faceDirection;

    #endif

    normal = normalize( normalMatrix * normal );

  // COMPAT: pre-r151
  // USE_NORMALMAP_TANGENTSPACE used to be TANGENTSPACE_NORMALMAP in pre-r151
  #elif defined( USE_NORMALMAP_TANGENTSPACE ) || defined( TANGENTSPACE_NORMALMAP )

    vec3 mapN = texture2D( normalMap, normalMapUv ).xyz * 2.0 - 1.0;
    mapN.xy *= normalScale;

    // COMPAT: pre-r151
    #if THREE_VRM_THREE_REVISION >= 151 || defined( USE_TANGENT )

      normal = normalize( tbn * mapN );

    #else

      normal = perturbNormal2Arb( uv, -vViewPosition, normal, mapN, faceDirection );

    #endif

  #endif

  // #include <emissivemap_fragment>
  #ifdef USE_EMISSIVEMAP
    vec2 emissiveMapUv = ( emissiveMapUvTransform * vec3( uv, 1 ) ).xy;
    totalEmissiveRadiance *= texture2D( emissiveMap, emissiveMapUv ).rgb;
  #endif

  #ifdef DEBUG_NORMAL
    gl_FragColor = vec4( 0.5 + 0.5 * normal, 1.0 );
    return;
  #endif

  // -- MToon: lighting --------------------------------------------------------
  // accumulation
  // #include <lights_phong_fragment>
  MToonMaterial material;

  material.diffuseColor = diffuseColor.rgb;

  material.shadeColor = shadeColorFactor;
  #ifdef USE_SHADEMULTIPLYTEXTURE
    vec2 shadeMultiplyTextureUv = ( shadeMultiplyTextureUvTransform * vec3( uv, 1 ) ).xy;
    material.shadeColor *= texture2D( shadeMultiplyTexture, shadeMultiplyTextureUv ).rgb;
  #endif

  #if ( defined( USE_COLOR ) && !defined( IGNORE_VERTEX_COLOR ) )
    material.shadeColor.rgb *= vColor;
  #endif

  material.shadingShift = shadingShiftFactor;
  #ifdef USE_SHADINGSHIFTTEXTURE
    vec2 shadingShiftTextureUv = ( shadingShiftTextureUvTransform * vec3( uv, 1 ) ).xy;
    material.shadingShift += texture2D( shadingShiftTexture, shadingShiftTextureUv ).r * shadingShiftTextureScale;
  #endif

  // #include <lights_fragment_begin>

  // MToon Specific changes:
  // Since we want to take shadows into account of shading instead of irradiance,
  // we had to modify the codes that multiplies the results of shadowmap into color of direct lights.

  // COMPAT: pre-r156 uses a struct GeometricContext
  #if THREE_VRM_THREE_REVISION >= 157
    vec3 geometryPosition = - vViewPosition;
    vec3 geometryNormal = normal;
    vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );

    vec3 geometryClearcoatNormal;

    #ifdef USE_CLEARCOAT

      geometryClearcoatNormal = clearcoatNormal;

    #endif
  #else
    GeometricContext geometry;

    geometry.position = - vViewPosition;
    geometry.normal = normal;
    geometry.viewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );

    #ifdef USE_CLEARCOAT

      geometry.clearcoatNormal = clearcoatNormal;

    #endif
  #endif

  IncidentLight directLight;

  // since these variables will be used in unrolled loop, we have to define in prior
  float shadow;

  #if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )

    PointLight pointLight;
    #if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
    PointLightShadow pointLightShadow;
    #endif

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {

      pointLight = pointLights[ i ];

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        getPointLightInfo( pointLight, geometryPosition, directLight );
      #else
        getPointLightInfo( pointLight, geometry, directLight );
      #endif

      shadow = 1.0;
      #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
      pointLightShadow = pointLightShadows[ i ];
      // COMPAT: pre-r166
      // r166 introduced shadowIntensity
      #if THREE_VRM_THREE_REVISION >= 166
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
      #else
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
      #endif
      #endif

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, shadow, reflectedLight );
      #else
        RE_Direct( directLight, geometry, material, shadow, reflectedLight );
      #endif

    }
    #pragma unroll_loop_end

  #endif

  #if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )

    SpotLight spotLight;
    // COMPAT: pre-r144 uses NUM_SPOT_LIGHT_SHADOWS, r144+ uses NUM_SPOT_LIGHT_COORDS
    #if THREE_VRM_THREE_REVISION >= 144
      #if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_COORDS > 0
      SpotLightShadow spotLightShadow;
      #endif
    #elif defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
    SpotLightShadow spotLightShadow;
    #endif

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {

      spotLight = spotLights[ i ];

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        getSpotLightInfo( spotLight, geometryPosition, directLight );
      #else
        getSpotLightInfo( spotLight, geometry, directLight );
      #endif

      shadow = 1.0;
      // COMPAT: pre-r144 uses NUM_SPOT_LIGHT_SHADOWS and vSpotShadowCoord, r144+ uses NUM_SPOT_LIGHT_COORDS and vSpotLightCoord
      // COMPAT: pre-r166 does not have shadowIntensity, r166+ has shadowIntensity
      #if THREE_VRM_THREE_REVISION >= 166
        #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_COORDS )
        spotLightShadow = spotLightShadows[ i ];
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
        #endif
      #elif THREE_VRM_THREE_REVISION >= 144
        #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_COORDS )
        spotLightShadow = spotLightShadows[ i ];
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
        #endif
      #elif defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
      spotLightShadow = spotLightShadows[ i ];
      shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotShadowCoord[ i ] ) : 1.0;
      #endif

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, shadow, reflectedLight );
      #else
        RE_Direct( directLight, geometry, material, shadow, reflectedLight );
      #endif

    }
    #pragma unroll_loop_end

  #endif

  #if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )

    DirectionalLight directionalLight;
    #if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
    DirectionalLightShadow directionalLightShadow;
    #endif

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {

      directionalLight = directionalLights[ i ];

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        getDirectionalLightInfo( directionalLight, directLight );
      #else
        getDirectionalLightInfo( directionalLight, geometry, directLight );
      #endif

      shadow = 1.0;
      #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
      directionalLightShadow = directionalLightShadows[ i ];
      // COMPAT: pre-r166
      // r166 introduced shadowIntensity
      #if THREE_VRM_THREE_REVISION >= 166
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
      #else
        shadow = all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
      #endif
      #endif

      // COMPAT: pre-r156 uses a struct GeometricContext
      #if THREE_VRM_THREE_REVISION >= 157
        RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, shadow, reflectedLight );
      #else
        RE_Direct( directLight, geometry, material, shadow, reflectedLight );
      #endif

    }
    #pragma unroll_loop_end

  #endif

  // #if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )

  //   RectAreaLight rectAreaLight;

  //   #pragma unroll_loop_start
  //   for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {

  //     rectAreaLight = rectAreaLights[ i ];
  //     RE_Direct_RectArea( rectAreaLight, geometry, material, reflectedLight );

  //   }
  //   #pragma unroll_loop_end

  // #endif

  #if defined( RE_IndirectDiffuse )

    vec3 iblIrradiance = vec3( 0.0 );

    vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );

    // COMPAT: pre-r156 uses a struct GeometricContext
    // COMPAT: pre-r156 doesn't have a define USE_LIGHT_PROBES
    #if THREE_VRM_THREE_REVISION >= 157
      #if defined( USE_LIGHT_PROBES )
        irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
      #endif
    #else
      irradiance += getLightProbeIrradiance( lightProbe, geometry.normal );
    #endif

    #if ( NUM_HEMI_LIGHTS > 0 )

      #pragma unroll_loop_start
      for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {

        // COMPAT: pre-r156 uses a struct GeometricContext
        #if THREE_VRM_THREE_REVISION >= 157
          irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
        #else
          irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry.normal );
        #endif

      }
      #pragma unroll_loop_end

    #endif

  #endif

  // #if defined( RE_IndirectSpecular )

  //   vec3 radiance = vec3( 0.0 );
  //   vec3 clearcoatRadiance = vec3( 0.0 );

  // #endif

  #include <lights_fragment_maps>
  #include <lights_fragment_end>

  // modulation
  #include <aomap_fragment>

  vec3 col = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;

  #ifdef DEBUG_LITSHADERATE
    gl_FragColor = vec4( col, diffuseColor.a );
    postCorrection();
    return;
  #endif

  // -- MToon: rim lighting -----------------------------------------
  vec3 viewDir = normalize( vViewPosition );

  #ifndef PHYSICALLY_CORRECT_LIGHTS
    reflectedLight.directSpecular /= PI;
  #endif
  vec3 rimMix = mix( vec3( 1.0 ), reflectedLight.directSpecular, rimLightingMixFactor );

  vec3 rim = parametricRimColorFactor * pow( saturate( 1.0 - dot( viewDir, normal ) + parametricRimLiftFactor ), parametricRimFresnelPowerFactor );

  #ifdef USE_MATCAPTEXTURE
    {
      vec3 x = normalize( vec3( viewDir.z, 0.0, -viewDir.x ) );
      vec3 y = cross( viewDir, x ); // guaranteed to be normalized
      vec2 sphereUv = 0.5 + 0.5 * vec2( dot( x, normal ), -dot( y, normal ) );
      sphereUv = ( matcapTextureUvTransform * vec3( sphereUv, 1 ) ).xy;
      vec3 matcap = texture2D( matcapTexture, sphereUv ).rgb;
      rim += matcapFactor * matcap;
    }
  #endif

  #ifdef USE_RIMMULTIPLYTEXTURE
    vec2 rimMultiplyTextureUv = ( rimMultiplyTextureUvTransform * vec3( uv, 1 ) ).xy;
    rim *= texture2D( rimMultiplyTexture, rimMultiplyTextureUv ).rgb;
  #endif

  col += rimMix * rim;

  // -- MToon: Emission --------------------------------------------------------
  col += totalEmissiveRadiance;

  // #include <envmap_fragment>

  // -- Almost done! -----------------------------------------------------------
  #if defined( OUTLINE )
    col = outlineColorFactor.rgb * mix( vec3( 1.0 ), col, outlineLightingMixFactor );
  #endif

  #ifdef OPAQUE
    diffuseColor.a = 1.0;
  #endif

  gl_FragColor = vec4( col, diffuseColor.a );
  postCorrection();
}
`,En={None:`none`,Normal:`normal`,LitShadeRate:`litShadeRate`,UV:`uv`},Dn={None:`none`,WorldCoordinates:`worldCoordinates`,ScreenCoordinates:`screenCoordinates`};function On(e){return e.colorSpace}var kn=class extends e{constructor(e={}){super({vertexShader:wn,fragmentShader:Tn}),this.uvAnimationScrollXSpeedFactor=0,this.uvAnimationScrollYSpeedFactor=0,this.uvAnimationRotationSpeedFactor=0,this.fog=!0,this.normalMapType=0,this._ignoreVertexColor=!0,this._v0CompatShade=!1,this._debugMode=En.None,this._outlineWidthMode=Dn.None,this._isOutline=!1,e.transparentWithZWrite&&(e.depthWrite=!0),delete e.transparentWithZWrite,e.fog=!0,e.lights=!0,e.clipping=!0,this.uniforms=o.merge([x.common,x.normalmap,x.emissivemap,x.fog,x.lights,{litFactor:{value:new S(1,1,1)},mapUvTransform:{value:new t},colorAlpha:{value:1},normalMapUvTransform:{value:new t},shadeColorFactor:{value:new S(0,0,0)},shadeMultiplyTexture:{value:null},shadeMultiplyTextureUvTransform:{value:new t},shadingShiftFactor:{value:0},shadingShiftTexture:{value:null},shadingShiftTextureUvTransform:{value:new t},shadingShiftTextureScale:{value:1},shadingToonyFactor:{value:.9},giEqualizationFactor:{value:.9},matcapFactor:{value:new S(1,1,1)},matcapTexture:{value:null},matcapTextureUvTransform:{value:new t},parametricRimColorFactor:{value:new S(0,0,0)},rimMultiplyTexture:{value:null},rimMultiplyTextureUvTransform:{value:new t},rimLightingMixFactor:{value:1},parametricRimFresnelPowerFactor:{value:5},parametricRimLiftFactor:{value:0},emissive:{value:new S(0,0,0)},emissiveIntensity:{value:1},emissiveMapUvTransform:{value:new t},outlineWidthMultiplyTexture:{value:null},outlineWidthMultiplyTextureUvTransform:{value:new t},outlineWidthFactor:{value:0},outlineColorFactor:{value:new S(0,0,0)},outlineLightingMixFactor:{value:1},uvAnimationMaskTexture:{value:null},uvAnimationMaskTextureUvTransform:{value:new t},uvAnimationScrollXOffset:{value:0},uvAnimationScrollYOffset:{value:0},uvAnimationRotationPhase:{value:0}},e.uniforms??{}]),this.setValues(e),this._uploadUniformsWorkaround(),this.customProgramCacheKey=()=>[...Object.entries(this._generateDefines()).map(([e,t])=>`${e}:${t}`),this.matcapTexture?`matcapTextureColorSpace:${On(this.matcapTexture)}`:``,this.shadeMultiplyTexture?`shadeMultiplyTextureColorSpace:${On(this.shadeMultiplyTexture)}`:``,this.rimMultiplyTexture?`rimMultiplyTextureColorSpace:${On(this.rimMultiplyTexture)}`:``].join(`,`),this.onBeforeCompile=e=>{let t=Object.entries(xn(xn({},this._generateDefines()),this.defines)).filter(([e,t])=>!!t).map(([e,t])=>`#define ${e} ${t}`).join(`
`)+`
`;e.vertexShader=t+e.vertexShader,e.fragmentShader=t+e.fragmentShader}}get color(){return this.uniforms.litFactor.value}set color(e){this.uniforms.litFactor.value=e}get map(){return this.uniforms.map.value}set map(e){this.uniforms.map.value=e}get normalMap(){return this.uniforms.normalMap.value}set normalMap(e){this.uniforms.normalMap.value=e}get normalScale(){return this.uniforms.normalScale.value}set normalScale(e){this.uniforms.normalScale.value=e}get emissive(){return this.uniforms.emissive.value}set emissive(e){this.uniforms.emissive.value=e}get emissiveIntensity(){return this.uniforms.emissiveIntensity.value}set emissiveIntensity(e){this.uniforms.emissiveIntensity.value=e}get emissiveMap(){return this.uniforms.emissiveMap.value}set emissiveMap(e){this.uniforms.emissiveMap.value=e}get shadeColorFactor(){return this.uniforms.shadeColorFactor.value}set shadeColorFactor(e){this.uniforms.shadeColorFactor.value=e}get shadeMultiplyTexture(){return this.uniforms.shadeMultiplyTexture.value}set shadeMultiplyTexture(e){this.uniforms.shadeMultiplyTexture.value=e}get shadingShiftFactor(){return this.uniforms.shadingShiftFactor.value}set shadingShiftFactor(e){this.uniforms.shadingShiftFactor.value=e}get shadingShiftTexture(){return this.uniforms.shadingShiftTexture.value}set shadingShiftTexture(e){this.uniforms.shadingShiftTexture.value=e}get shadingShiftTextureScale(){return this.uniforms.shadingShiftTextureScale.value}set shadingShiftTextureScale(e){this.uniforms.shadingShiftTextureScale.value=e}get shadingToonyFactor(){return this.uniforms.shadingToonyFactor.value}set shadingToonyFactor(e){this.uniforms.shadingToonyFactor.value=e}get giEqualizationFactor(){return this.uniforms.giEqualizationFactor.value}set giEqualizationFactor(e){this.uniforms.giEqualizationFactor.value=e}get matcapFactor(){return this.uniforms.matcapFactor.value}set matcapFactor(e){this.uniforms.matcapFactor.value=e}get matcapTexture(){return this.uniforms.matcapTexture.value}set matcapTexture(e){this.uniforms.matcapTexture.value=e}get parametricRimColorFactor(){return this.uniforms.parametricRimColorFactor.value}set parametricRimColorFactor(e){this.uniforms.parametricRimColorFactor.value=e}get rimMultiplyTexture(){return this.uniforms.rimMultiplyTexture.value}set rimMultiplyTexture(e){this.uniforms.rimMultiplyTexture.value=e}get rimLightingMixFactor(){return this.uniforms.rimLightingMixFactor.value}set rimLightingMixFactor(e){this.uniforms.rimLightingMixFactor.value=e}get parametricRimFresnelPowerFactor(){return this.uniforms.parametricRimFresnelPowerFactor.value}set parametricRimFresnelPowerFactor(e){this.uniforms.parametricRimFresnelPowerFactor.value=e}get parametricRimLiftFactor(){return this.uniforms.parametricRimLiftFactor.value}set parametricRimLiftFactor(e){this.uniforms.parametricRimLiftFactor.value=e}get outlineWidthMultiplyTexture(){return this.uniforms.outlineWidthMultiplyTexture.value}set outlineWidthMultiplyTexture(e){this.uniforms.outlineWidthMultiplyTexture.value=e}get outlineWidthFactor(){return this.uniforms.outlineWidthFactor.value}set outlineWidthFactor(e){this.uniforms.outlineWidthFactor.value=e}get outlineColorFactor(){return this.uniforms.outlineColorFactor.value}set outlineColorFactor(e){this.uniforms.outlineColorFactor.value=e}get outlineLightingMixFactor(){return this.uniforms.outlineLightingMixFactor.value}set outlineLightingMixFactor(e){this.uniforms.outlineLightingMixFactor.value=e}get uvAnimationMaskTexture(){return this.uniforms.uvAnimationMaskTexture.value}set uvAnimationMaskTexture(e){this.uniforms.uvAnimationMaskTexture.value=e}get uvAnimationScrollXOffset(){return this.uniforms.uvAnimationScrollXOffset.value}set uvAnimationScrollXOffset(e){this.uniforms.uvAnimationScrollXOffset.value=e}get uvAnimationScrollYOffset(){return this.uniforms.uvAnimationScrollYOffset.value}set uvAnimationScrollYOffset(e){this.uniforms.uvAnimationScrollYOffset.value=e}get uvAnimationRotationPhase(){return this.uniforms.uvAnimationRotationPhase.value}set uvAnimationRotationPhase(e){this.uniforms.uvAnimationRotationPhase.value=e}get ignoreVertexColor(){return this._ignoreVertexColor}set ignoreVertexColor(e){this._ignoreVertexColor=e,this.needsUpdate=!0}get v0CompatShade(){return this._v0CompatShade}set v0CompatShade(e){this._v0CompatShade=e,this.needsUpdate=!0}get debugMode(){return this._debugMode}set debugMode(e){this._debugMode=e,this.needsUpdate=!0}get outlineWidthMode(){return this._outlineWidthMode}set outlineWidthMode(e){this._outlineWidthMode=e,this.needsUpdate=!0}get isOutline(){return this._isOutline}set isOutline(e){this._isOutline=e,this.needsUpdate=!0}get isMToonMaterial(){return!0}update(e){this._uploadUniformsWorkaround(),this._updateUVAnimation(e)}copy(e){return super.copy(e),this.map=e.map,this.normalMap=e.normalMap,this.emissiveMap=e.emissiveMap,this.shadeMultiplyTexture=e.shadeMultiplyTexture,this.shadingShiftTexture=e.shadingShiftTexture,this.matcapTexture=e.matcapTexture,this.rimMultiplyTexture=e.rimMultiplyTexture,this.outlineWidthMultiplyTexture=e.outlineWidthMultiplyTexture,this.uvAnimationMaskTexture=e.uvAnimationMaskTexture,this.normalMapType=e.normalMapType,this.uvAnimationScrollXSpeedFactor=e.uvAnimationScrollXSpeedFactor,this.uvAnimationScrollYSpeedFactor=e.uvAnimationScrollYSpeedFactor,this.uvAnimationRotationSpeedFactor=e.uvAnimationRotationSpeedFactor,this.ignoreVertexColor=e.ignoreVertexColor,this.v0CompatShade=e.v0CompatShade,this.debugMode=e.debugMode,this.outlineWidthMode=e.outlineWidthMode,this.isOutline=e.isOutline,this.needsUpdate=!0,this}_updateUVAnimation(e){this.uniforms.uvAnimationScrollXOffset.value+=e*this.uvAnimationScrollXSpeedFactor,this.uniforms.uvAnimationScrollYOffset.value+=e*this.uvAnimationScrollYSpeedFactor,this.uniforms.uvAnimationRotationPhase.value+=e*this.uvAnimationRotationSpeedFactor,this.uniforms.alphaTest.value=this.alphaTest,this.uniformsNeedUpdate=!0}_uploadUniformsWorkaround(){this.uniforms.opacity.value=this.opacity,this._updateTextureMatrix(this.uniforms.map,this.uniforms.mapUvTransform),this._updateTextureMatrix(this.uniforms.normalMap,this.uniforms.normalMapUvTransform),this._updateTextureMatrix(this.uniforms.emissiveMap,this.uniforms.emissiveMapUvTransform),this._updateTextureMatrix(this.uniforms.shadeMultiplyTexture,this.uniforms.shadeMultiplyTextureUvTransform),this._updateTextureMatrix(this.uniforms.shadingShiftTexture,this.uniforms.shadingShiftTextureUvTransform),this._updateTextureMatrix(this.uniforms.matcapTexture,this.uniforms.matcapTextureUvTransform),this._updateTextureMatrix(this.uniforms.rimMultiplyTexture,this.uniforms.rimMultiplyTextureUvTransform),this._updateTextureMatrix(this.uniforms.outlineWidthMultiplyTexture,this.uniforms.outlineWidthMultiplyTextureUvTransform),this._updateTextureMatrix(this.uniforms.uvAnimationMaskTexture,this.uniforms.uvAnimationMaskTextureUvTransform),this.uniformsNeedUpdate=!0}_generateDefines(){let e=this.outlineWidthMultiplyTexture!==null,t=this.map!==null||this.normalMap!==null||this.emissiveMap!==null||this.shadeMultiplyTexture!==null||this.shadingShiftTexture!==null||this.rimMultiplyTexture!==null||this.uvAnimationMaskTexture!==null;return{THREE_VRM_THREE_REVISION:181,OUTLINE:this._isOutline,MTOON_USE_UV:e||t,MTOON_UVS_VERTEX_ONLY:e&&!t,V0_COMPAT_SHADE:this._v0CompatShade,USE_SHADEMULTIPLYTEXTURE:this.shadeMultiplyTexture!==null,USE_SHADINGSHIFTTEXTURE:this.shadingShiftTexture!==null,USE_MATCAPTEXTURE:this.matcapTexture!==null,USE_RIMMULTIPLYTEXTURE:this.rimMultiplyTexture!==null,USE_OUTLINEWIDTHMULTIPLYTEXTURE:this._isOutline&&this.outlineWidthMultiplyTexture!==null,USE_UVANIMATIONMASKTEXTURE:this.uvAnimationMaskTexture!==null,IGNORE_VERTEX_COLOR:this._ignoreVertexColor===!0,DEBUG_NORMAL:this._debugMode===`normal`,DEBUG_LITSHADERATE:this._debugMode===`litShadeRate`,DEBUG_UV:this._debugMode===`uv`,OUTLINE_WIDTH_SCREEN:this._isOutline&&this._outlineWidthMode===Dn.ScreenCoordinates}}_updateTextureMatrix(e,t){e.value&&(e.value.matrixAutoUpdate&&e.value.updateMatrix(),t.value.copy(e.value.matrix))}},An=new Set([`1.0`,`1.0-beta`]),jn=class e{get name(){return e.EXTENSION_NAME}constructor(e,t={}){this.parser=e,this.materialType=t.materialType??kn,this.renderOrderOffset=t.renderOrderOffset??0,this.v0CompatShade=t.v0CompatShade??!1,this.debugMode=t.debugMode??`none`,this._mToonMaterialSet=new Set}beforeRoot(){return H(this,null,function*(){this._removeUnlitExtensionIfMToonExists()})}afterRoot(e){return H(this,null,function*(){e.userData.vrmMToonMaterials=Array.from(this._mToonMaterialSet)})}getMaterialType(e){return this._getMToonExtension(e)?this.materialType:null}extendMaterialParams(e,t){let n=this._getMToonExtension(e);return n?this._extendMaterialParams(n,t):null}loadMesh(e){return H(this,null,function*(){let t=this.parser,n=t.json.meshes?.[e];if(n==null)throw Error(`MToonMaterialLoaderPlugin: Attempt to use meshes[${e}] of glTF but the mesh doesn't exist`);let r=n.primitives,i=yield t.loadMesh(e);if(r.length===1){let e=i,t=r[0].material;t!=null&&this._setupPrimitive(e,t)}else{let e=i;for(let t=0;t<r.length;t++){let n=e.children[t],i=r[t].material;i!=null&&this._setupPrimitive(n,i)}}return i})}_removeUnlitExtensionIfMToonExists(){this.parser.json.materials?.map((e,t)=>{this._getMToonExtension(t)&&e.extensions?.KHR_materials_unlit&&delete e.extensions.KHR_materials_unlit})}_getMToonExtension(t){let n=this.parser.json.materials?.[t];if(n==null)return;let r=n.extensions?.[e.EXTENSION_NAME];if(r==null)return;let i=r.specVersion;if(An.has(i))return r}_extendMaterialParams(e,t){return H(this,null,function*(){delete t.metalness,delete t.roughness;let n=new Cn(this.parser,t);n.assignPrimitive(`transparentWithZWrite`,e.transparentWithZWrite),n.assignColor(`shadeColorFactor`,e.shadeColorFactor),n.assignTexture(`shadeMultiplyTexture`,e.shadeMultiplyTexture,!0),n.assignPrimitive(`shadingShiftFactor`,e.shadingShiftFactor),n.assignTexture(`shadingShiftTexture`,e.shadingShiftTexture,!0),n.assignPrimitive(`shadingShiftTextureScale`,e.shadingShiftTexture?.scale),n.assignPrimitive(`shadingToonyFactor`,e.shadingToonyFactor),n.assignPrimitive(`giEqualizationFactor`,e.giEqualizationFactor),n.assignColor(`matcapFactor`,e.matcapFactor),n.assignTexture(`matcapTexture`,e.matcapTexture,!0),n.assignColor(`parametricRimColorFactor`,e.parametricRimColorFactor),n.assignTexture(`rimMultiplyTexture`,e.rimMultiplyTexture,!0),n.assignPrimitive(`rimLightingMixFactor`,e.rimLightingMixFactor),n.assignPrimitive(`parametricRimFresnelPowerFactor`,e.parametricRimFresnelPowerFactor),n.assignPrimitive(`parametricRimLiftFactor`,e.parametricRimLiftFactor),n.assignPrimitive(`outlineWidthMode`,e.outlineWidthMode),n.assignPrimitive(`outlineWidthFactor`,e.outlineWidthFactor),n.assignTexture(`outlineWidthMultiplyTexture`,e.outlineWidthMultiplyTexture,!1),n.assignColor(`outlineColorFactor`,e.outlineColorFactor),n.assignPrimitive(`outlineLightingMixFactor`,e.outlineLightingMixFactor),n.assignTexture(`uvAnimationMaskTexture`,e.uvAnimationMaskTexture,!1),n.assignPrimitive(`uvAnimationScrollXSpeedFactor`,e.uvAnimationScrollXSpeedFactor),n.assignPrimitive(`uvAnimationScrollYSpeedFactor`,e.uvAnimationScrollYSpeedFactor),n.assignPrimitive(`uvAnimationRotationSpeedFactor`,e.uvAnimationRotationSpeedFactor),n.assignPrimitive(`v0CompatShade`,this.v0CompatShade),n.assignPrimitive(`debugMode`,this.debugMode),yield n.pending})}_setupPrimitive(e,t){let n=this._getMToonExtension(t);n&&(e.renderOrder=this._parseRenderOrder(n)+this.renderOrderOffset,this._generateOutline(e),this._addToMaterialSet(e))}_shouldGenerateOutline(e){return typeof e.outlineWidthMode==`string`&&e.outlineWidthMode!==`none`&&typeof e.outlineWidthFactor==`number`&&e.outlineWidthFactor>0}_generateOutline(e){let t=e.material;if(!(t instanceof se)||!this._shouldGenerateOutline(t))return;e.material=[t];let n=t.clone();n.name+=` (Outline)`,n.isOutline=!0,n.side=1,e.material.push(n);let r=e.geometry,i=r.index?r.index.count:r.attributes.position.count/3;r.addGroup(0,i,0),r.addGroup(0,i,1)}_addToMaterialSet(e){let t=e.material,n=new Set;Array.isArray(t)?t.forEach(e=>n.add(e)):n.add(t);for(let e of n)this._mToonMaterialSet.add(e)}_parseRenderOrder(e){return(e.transparentWithZWrite?0:19)+(e.renderQueueOffsetNumber??0)}};jn.EXTENSION_NAME=`VRMC_materials_mtoon`;var Mn=jn,Nn=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())}),Pn=class e{get name(){return e.EXTENSION_NAME}constructor(e){this.parser=e}extendMaterialParams(e,t){return Nn(this,null,function*(){let n=this._getHDREmissiveMultiplierExtension(e);n!=null&&(t.emissiveIntensity=n.emissiveMultiplier)})}_getHDREmissiveMultiplierExtension(t){let n=this.parser.json.materials?.[t];if(n==null)return;let r=n.extensions?.[e.EXTENSION_NAME];if(r!=null)return r}};Pn.EXTENSION_NAME=`VRMC_materials_hdr_emissiveMultiplier`;var Fn=Pn,In=Object.defineProperty,Ln=Object.defineProperties,Rn=Object.getOwnPropertyDescriptors,zn=Object.getOwnPropertySymbols,Bn=Object.prototype.hasOwnProperty,Vn=Object.prototype.propertyIsEnumerable,Hn=(e,t,n)=>t in e?In(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,U=(e,t)=>{for(var n in t||={})Bn.call(t,n)&&Hn(e,n,t[n]);if(zn)for(var n of zn(t))Vn.call(t,n)&&Hn(e,n,t[n]);return e},Un=(e,t)=>Ln(e,Rn(t)),Wn=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())});function W(e){return e**2.2}var Gn=class{get name(){return`VRMMaterialsV0CompatPlugin`}constructor(e){this.parser=e,this._renderQueueMapTransparent=new Map,this._renderQueueMapTransparentZWrite=new Map;let t=this.parser.json;t.extensionsUsed=t.extensionsUsed??[],t.extensionsUsed.indexOf(`KHR_texture_transform`)===-1&&t.extensionsUsed.push(`KHR_texture_transform`)}beforeRoot(){return Wn(this,null,function*(){let e=this.parser.json,t=e.extensions?.VRM?.materialProperties;t&&(this._populateRenderQueueMap(t),t.forEach((t,n)=>{let r=e.materials?.[n];if(r!=null){if(t.shader===`VRM/MToon`){let i=this._parseV0MToonProperties(t,r);e.materials[n]=i}else if(t.shader?.startsWith(`VRM/Unlit`)){let i=this._parseV0UnlitProperties(t,r);e.materials[n]=i}else t.shader}}))})}_parseV0MToonProperties(e,t){let n=e.keywordMap?._ALPHABLEND_ON??!1,r=e.floatProperties?._ZWrite===1&&n,i=this._v0ParseRenderQueue(e),a=e.keywordMap?._ALPHATEST_ON??!1,o=n?`BLEND`:a?`MASK`:`OPAQUE`,s=a?e.floatProperties?._Cutoff??.5:void 0,c=(e.floatProperties?._CullMode??2)===0,u=this._portTextureTransform(e),d=(e.vectorProperties?._Color??[1,1,1,1]).map((e,t)=>t===3?e:W(e)),f=e.textureProperties?._MainTex,p=f==null?void 0:{index:f,extensions:U({},u)},m=e.floatProperties?._BumpScale??1,h=e.textureProperties?._BumpMap,g=h==null?void 0:{index:h,scale:m,extensions:U({},u)},_=(e.vectorProperties?._EmissionColor??[0,0,0,1]).map(W),v=e.textureProperties?._EmissionMap,ee=v==null?void 0:{index:v,extensions:U({},u)},te=(e.vectorProperties?._ShadeColor??[.97,.81,.86,1]).map(W),y=e.textureProperties?._ShadeTexture,b=y==null?void 0:{index:y,extensions:U({},u)},ne=e.floatProperties?._ShadeShift??0,x=e.floatProperties?._ShadeToony??.9;x=l.lerp(x,1,.5+.5*ne),ne=-ne-(1-x);let re=e.floatProperties?._IndirectLightIntensity??.1,ie=re?1-re:void 0,ae=e.textureProperties?._SphereAdd,S=ae==null?void 0:[1,1,1],oe=ae==null?void 0:{index:ae},C=e.floatProperties?._RimLightingMix??0,se=e.textureProperties?._RimTexture,ce=se==null?void 0:{index:se,extensions:U({},u)},le=(e.vectorProperties?._RimColor??[0,0,0,1]).map(W),ue=e.floatProperties?._RimFresnelPower??1,de=e.floatProperties?._RimLift??0,fe=[`none`,`worldCoordinates`,`screenCoordinates`][e.floatProperties?._OutlineWidthMode??0],w=e.floatProperties?._OutlineWidth??0;w=.01*w;let pe=e.textureProperties?._OutlineWidthTexture,me=pe==null?void 0:{index:pe,extensions:U({},u)},he=(e.vectorProperties?._OutlineColor??[0,0,0]).map(W),T=(e.floatProperties?._OutlineColorMode??0)===1?e.floatProperties?._OutlineLightingMix??1:0,E=e.textureProperties?._UvAnimMaskTexture,D=E==null?void 0:{index:E,extensions:U({},u)},ge=e.floatProperties?._UvAnimScrollX??0,O=e.floatProperties?._UvAnimScrollY??0;O!=null&&(O=-O);let _e=e.floatProperties?._UvAnimRotation??0,ve={specVersion:`1.0`,transparentWithZWrite:r,renderQueueOffsetNumber:i,shadeColorFactor:te,shadeMultiplyTexture:b,shadingShiftFactor:ne,shadingToonyFactor:x,giEqualizationFactor:ie,matcapFactor:S,matcapTexture:oe,rimLightingMixFactor:C,rimMultiplyTexture:ce,parametricRimColorFactor:le,parametricRimFresnelPowerFactor:ue,parametricRimLiftFactor:de,outlineWidthMode:fe,outlineWidthFactor:w,outlineWidthMultiplyTexture:me,outlineColorFactor:he,outlineLightingMixFactor:T,uvAnimationMaskTexture:D,uvAnimationScrollXSpeedFactor:ge,uvAnimationScrollYSpeedFactor:O,uvAnimationRotationSpeedFactor:_e};return Un(U({},t),{pbrMetallicRoughness:{baseColorFactor:d,baseColorTexture:p},normalTexture:g,emissiveTexture:ee,emissiveFactor:_,alphaMode:o,alphaCutoff:s,doubleSided:c,extensions:{VRMC_materials_mtoon:ve}})}_parseV0UnlitProperties(e,t){let n=e.shader===`VRM/UnlitTransparentZWrite`,r=e.shader===`VRM/UnlitTransparent`||n,i=this._v0ParseRenderQueue(e),a=e.shader===`VRM/UnlitCutout`,o=r?`BLEND`:a?`MASK`:`OPAQUE`,s=a?e.floatProperties?._Cutoff??.5:void 0,c=this._portTextureTransform(e),l=(e.vectorProperties?._Color??[1,1,1,1]).map(W),u=e.textureProperties?._MainTex,d=u==null?void 0:{index:u,extensions:U({},c)},f={specVersion:`1.0`,transparentWithZWrite:n,renderQueueOffsetNumber:i,shadeColorFactor:l,shadeMultiplyTexture:d};return Un(U({},t),{pbrMetallicRoughness:{baseColorFactor:l,baseColorTexture:d},alphaMode:o,alphaCutoff:s,extensions:{VRMC_materials_mtoon:f}})}_portTextureTransform(e){let t=e.vectorProperties?._MainTex;if(t==null)return{};let n=[t?.[0]??0,t?.[1]??0],r=[t?.[2]??1,t?.[3]??1];return n[1]=1-r[1]-n[1],{KHR_texture_transform:{offset:n,scale:r}}}_v0ParseRenderQueue(e){let t=e.shader===`VRM/UnlitTransparentZWrite`,n=e.keywordMap?._ALPHABLEND_ON!=null||e.shader===`VRM/UnlitTransparent`||t,r=e.floatProperties?._ZWrite===1||t,i=0;if(n){let t=e.renderQueue;t!=null&&(i=r?this._renderQueueMapTransparentZWrite.get(t):this._renderQueueMapTransparent.get(t))}return i}_populateRenderQueueMap(e){let t=new Set,n=new Set;e.forEach(e=>{let r=e.shader===`VRM/UnlitTransparentZWrite`,i=e.keywordMap?._ALPHABLEND_ON!=null||e.shader===`VRM/UnlitTransparent`||r,a=e.floatProperties?._ZWrite===1||r;if(i){let r=e.renderQueue;r!=null&&(a?n.add(r):t.add(r))}}),t.size,n.size,Array.from(t).sort().forEach((e,n)=>{let r=Math.min(Math.max(n-t.size+1,-9),0);this._renderQueueMapTransparent.set(e,r)}),Array.from(n).sort().forEach((e,t)=>{let n=Math.min(Math.max(t,0),9);this._renderQueueMapTransparentZWrite.set(e,n)})}},Kn=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())}),G=new i,qn=class extends b{constructor(e){super(),this._attrPosition=new m(new Float32Array([0,0,0,0,0,0]),3),this._attrPosition.setUsage(h);let t=new y;t.setAttribute(`position`,this._attrPosition);let n=new ce({color:16711935,depthTest:!1,depthWrite:!1});this._line=new oe(t,n),this.add(this._line),this.constraint=e}updateMatrixWorld(e){G.setFromMatrixPosition(this.constraint.destination.matrixWorld),this._attrPosition.setXYZ(0,G.x,G.y,G.z),this.constraint.source&&G.setFromMatrixPosition(this.constraint.source.matrixWorld),this._attrPosition.setXYZ(1,G.x,G.y,G.z),this._attrPosition.needsUpdate=!0,super.updateMatrixWorld(e)}};function Jn(e,t){return t.set(e.elements[12],e.elements[13],e.elements[14])}var Yn=new i,Xn=new i;function Zn(e,t){return e.decompose(Yn,t,Xn),t}function Qn(e){return e.invert?e.invert():e.inverse(),e}var $n=class{constructor(e,t){this.destination=e,this.source=t,this.weight=1}},er=new i,tr=new i,nr=new i,rr=new s,ir=new s,ar=new s,or=class extends $n{get aimAxis(){return this._aimAxis}set aimAxis(e){this._aimAxis=e,this._v3AimAxis.set(e===`PositiveX`?1:e===`NegativeX`?-1:0,e===`PositiveY`?1:e===`NegativeY`?-1:0,e===`PositiveZ`?1:e===`NegativeZ`?-1:0)}get dependencies(){let e=new Set([this.source]);return this.destination.parent&&e.add(this.destination.parent),e}constructor(e,t){super(e,t),this._aimAxis=`PositiveX`,this._v3AimAxis=new i(1,0,0),this._dstRestQuat=new s}setInitState(){this._dstRestQuat.copy(this.destination.quaternion)}update(){this.destination.updateWorldMatrix(!0,!1),this.source.updateWorldMatrix(!0,!1);let e=rr.identity(),t=ir.identity();this.destination.parent&&(Zn(this.destination.parent.matrixWorld,e),Qn(t.copy(e)));let n=er.copy(this._v3AimAxis).applyQuaternion(this._dstRestQuat).applyQuaternion(e),r=Jn(this.source.matrixWorld,tr).sub(Jn(this.destination.matrixWorld,nr)).normalize(),i=ar.setFromUnitVectors(n,r).premultiply(t).multiply(e).multiply(this._dstRestQuat);this.destination.quaternion.copy(this._dstRestQuat).slerp(i,this.weight)}};function sr(e,t){let n=[e],r=e.parent;for(;r!==null;)n.unshift(r),r=r.parent;n.forEach(e=>{t(e)})}var cr=class{constructor(){this._constraints=new Set,this._objectConstraintsMap=new Map}get constraints(){return this._constraints}addConstraint(e){this._constraints.add(e);let t=this._objectConstraintsMap.get(e.destination);t??(t=new Set,this._objectConstraintsMap.set(e.destination,t)),t.add(e)}deleteConstraint(e){this._constraints.delete(e),this._objectConstraintsMap.get(e.destination).delete(e)}setInitState(){let e=new Set,t=new Set;for(let n of this._constraints)this._processConstraint(n,e,t,e=>e.setInitState())}update(){let e=new Set,t=new Set;for(let n of this._constraints)this._processConstraint(n,e,t,e=>e.update())}_processConstraint(e,t,n,r){if(n.has(e))return;if(t.has(e))throw Error(`VRMNodeConstraintManager: Circular dependency detected while updating constraints`);t.add(e);let i=e.dependencies;for(let e of i)sr(e,e=>{let i=this._objectConstraintsMap.get(e);if(i)for(let e of i)this._processConstraint(e,t,n,r)});r(e),n.add(e)}},lr=new s,ur=new s,dr=class extends $n{get dependencies(){return new Set([this.source])}constructor(e,t){super(e,t),this._dstRestQuat=new s,this._invSrcRestQuat=new s}setInitState(){this._dstRestQuat.copy(this.destination.quaternion),Qn(this._invSrcRestQuat.copy(this.source.quaternion))}update(){let e=lr.copy(this._invSrcRestQuat).multiply(this.source.quaternion),t=ur.copy(this._dstRestQuat).multiply(e);this.destination.quaternion.copy(this._dstRestQuat).slerp(t,this.weight)}},fr=new i,pr=new s,mr=new s,hr=class extends $n{get rollAxis(){return this._rollAxis}set rollAxis(e){this._rollAxis=e,this._v3RollAxis.set(+(e===`X`),+(e===`Y`),+(e===`Z`))}get dependencies(){return new Set([this.source])}constructor(e,t){super(e,t),this._rollAxis=`X`,this._v3RollAxis=new i(1,0,0),this._dstRestQuat=new s,this._invDstRestQuat=new s,this._invSrcRestQuatMulDstRestQuat=new s}setInitState(){this._dstRestQuat.copy(this.destination.quaternion),Qn(this._invDstRestQuat.copy(this._dstRestQuat)),Qn(this._invSrcRestQuatMulDstRestQuat.copy(this.source.quaternion)).multiply(this._dstRestQuat)}update(){let e=pr.copy(this._invDstRestQuat).multiply(this.source.quaternion).multiply(this._invSrcRestQuatMulDstRestQuat),t=fr.copy(this._v3RollAxis).applyQuaternion(e),n=mr.setFromUnitVectors(t,this._v3RollAxis).premultiply(this._dstRestQuat).multiply(e);this.destination.quaternion.copy(this._dstRestQuat).slerp(n,this.weight)}},gr=new Set([`1.0`,`1.0-beta`]),_r=class e{get name(){return e.EXTENSION_NAME}constructor(e,t){this.parser=e,this.helperRoot=t?.helperRoot}afterRoot(e){return Kn(this,null,function*(){e.userData.vrmNodeConstraintManager=yield this._import(e)})}_import(t){return Kn(this,null,function*(){let n=this.parser.json;if(n.extensionsUsed?.indexOf(e.EXTENSION_NAME)===-1)return null;let r=new cr,i=yield this.parser.getDependencies(`node`);return i.forEach((t,a)=>{let o=n.nodes[a]?.extensions?.[e.EXTENSION_NAME];if(o==null)return;let s=o.specVersion;if(!gr.has(s))return;let c=o.constraint;if(c.roll!=null){let e=this._importRollConstraint(t,i,c.roll);r.addConstraint(e)}else if(c.aim!=null){let e=this._importAimConstraint(t,i,c.aim);r.addConstraint(e)}else if(c.rotation!=null){let e=this._importRotationConstraint(t,i,c.rotation);r.addConstraint(e)}}),t.scene.updateMatrixWorld(),r.setInitState(),r})}_importRollConstraint(e,t,n){let{source:r,rollAxis:i,weight:a}=n,o=t[r],s=new hr(e,o);if(i!=null&&(s.rollAxis=i),a!=null&&(s.weight=a),this.helperRoot){let e=new qn(s);this.helperRoot.add(e)}return s}_importAimConstraint(e,t,n){let{source:r,aimAxis:i,weight:a}=n,o=t[r],s=new or(e,o);if(i!=null&&(s.aimAxis=i),a!=null&&(s.weight=a),this.helperRoot){let e=new qn(s);this.helperRoot.add(e)}return s}_importRotationConstraint(e,t,n){let{source:r,weight:i}=n,a=t[r],o=new dr(e,a);if(i!=null&&(o.weight=i),this.helperRoot){let e=new qn(o);this.helperRoot.add(e)}return o}};_r.EXTENSION_NAME=`VRMC_node_constraint`;var vr=_r,yr=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())}),br=class{},xr=new i,K=new i,Sr=class extends br{get type(){return`capsule`}constructor(e){super(),this.offset=e?.offset??new i(0,0,0),this.tail=e?.tail??new i(0,0,0),this.radius=e?.radius??0,this.inside=e?.inside??!1}calculateCollision(e,t,n,r){xr.setFromMatrixPosition(e),K.subVectors(this.tail,this.offset).applyMatrix4(e),K.sub(xr);let i=K.lengthSq();r.copy(t).sub(xr);let a=K.dot(r);a<=0||(i<=a||K.multiplyScalar(a/i),r.sub(K));let o=r.length(),s=this.inside?this.radius-n-o:o-n-this.radius;return s<0&&(r.multiplyScalar(1/o),this.inside&&r.negate()),s}},Cr=new i,wr=new t,Tr=class extends br{get type(){return`plane`}constructor(e){super(),this.offset=e?.offset??new i(0,0,0),this.normal=e?.normal??new i(0,0,1)}calculateCollision(e,t,n,r){r.setFromMatrixPosition(e),r.negate().add(t),wr.getNormalMatrix(e),Cr.copy(this.normal).applyNormalMatrix(wr).normalize();let i=r.dot(Cr)-n;return r.copy(Cr),i}},Er=new i,Dr=class extends br{get type(){return`sphere`}constructor(e){super(),this.offset=e?.offset??new i(0,0,0),this.radius=e?.radius??0,this.inside=e?.inside??!1}calculateCollision(e,t,n,r){r.subVectors(t,Er.setFromMatrixPosition(e));let i=r.length(),a=this.inside?this.radius-n-i:i-n-this.radius;return a<0&&(r.multiplyScalar(1/i),this.inside&&r.negate()),a}},q=new i,Or=class extends y{constructor(e){super(),this.worldScale=1,this._currentRadius=0,this._currentOffset=new i,this._currentTail=new i,this._shape=e,this._attrPos=new m(new Float32Array(396),3),this.setAttribute(`position`,this._attrPos),this._attrIndex=new m(new Uint16Array(264),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1,t=this._shape.radius/this.worldScale;this._currentRadius!==t&&(this._currentRadius=t,e=!0),this._currentOffset.equals(this._shape.offset)||(this._currentOffset.copy(this._shape.offset),e=!0);let n=q.copy(this._shape.tail).divideScalar(this.worldScale);this._currentTail.distanceToSquared(n)>1e-10&&(this._currentTail.copy(n),e=!0),e&&this._buildPosition()}_buildPosition(){q.copy(this._currentTail).sub(this._currentOffset);let e=q.length()/this._currentRadius;for(let t=0;t<=16;t++){let n=t/16*Math.PI;this._attrPos.setXYZ(t,-Math.sin(n),-Math.cos(n),0),this._attrPos.setXYZ(17+t,e+Math.sin(n),Math.cos(n),0),this._attrPos.setXYZ(34+t,-Math.sin(n),0,-Math.cos(n)),this._attrPos.setXYZ(51+t,e+Math.sin(n),0,Math.cos(n))}for(let t=0;t<32;t++){let n=t/16*Math.PI;this._attrPos.setXYZ(68+t,0,Math.sin(n),Math.cos(n)),this._attrPos.setXYZ(100+t,e,Math.sin(n),Math.cos(n))}let t=Math.atan2(q.y,Math.sqrt(q.x*q.x+q.z*q.z)),n=-Math.atan2(q.z,q.x);this.rotateZ(t),this.rotateY(n),this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentOffset.x,this._currentOffset.y,this._currentOffset.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<34;e++){let t=(e+1)%34;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(68+e*2,34+e,34+t)}for(let e=0;e<32;e++){let t=(e+1)%32;this._attrIndex.setXY(136+e*2,68+e,68+t),this._attrIndex.setXY(200+e*2,100+e,100+t)}this._attrIndex.needsUpdate=!0}},kr=class extends y{constructor(e){super(),this.worldScale=1,this._currentOffset=new i,this._currentNormal=new i,this._shape=e,this._attrPos=new m(new Float32Array(18),3),this.setAttribute(`position`,this._attrPos),this._attrIndex=new m(new Uint16Array(10),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1;this._currentOffset.equals(this._shape.offset)||(this._currentOffset.copy(this._shape.offset),e=!0),this._currentNormal.equals(this._shape.normal)||(this._currentNormal.copy(this._shape.normal),e=!0),e&&this._buildPosition()}_buildPosition(){this._attrPos.setXYZ(0,-.5,-.5,0),this._attrPos.setXYZ(1,.5,-.5,0),this._attrPos.setXYZ(2,.5,.5,0),this._attrPos.setXYZ(3,-.5,.5,0),this._attrPos.setXYZ(4,0,0,0),this._attrPos.setXYZ(5,0,0,.25),this.translate(this._currentOffset.x,this._currentOffset.y,this._currentOffset.z),this.lookAt(this._currentNormal),this._attrPos.needsUpdate=!0}_buildIndex(){this._attrIndex.setXY(0,0,1),this._attrIndex.setXY(2,1,2),this._attrIndex.setXY(4,2,3),this._attrIndex.setXY(6,3,0),this._attrIndex.setXY(8,4,5),this._attrIndex.needsUpdate=!0}},Ar=class extends y{constructor(e){super(),this.worldScale=1,this._currentRadius=0,this._currentOffset=new i,this._shape=e,this._attrPos=new m(new Float32Array(288),3),this.setAttribute(`position`,this._attrPos),this._attrIndex=new m(new Uint16Array(192),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1,t=this._shape.radius/this.worldScale;this._currentRadius!==t&&(this._currentRadius=t,e=!0),this._currentOffset.equals(this._shape.offset)||(this._currentOffset.copy(this._shape.offset),e=!0),e&&this._buildPosition()}_buildPosition(){for(let e=0;e<32;e++){let t=e/16*Math.PI;this._attrPos.setXYZ(e,Math.cos(t),Math.sin(t),0),this._attrPos.setXYZ(32+e,0,Math.cos(t),Math.sin(t)),this._attrPos.setXYZ(64+e,Math.sin(t),0,Math.cos(t))}this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentOffset.x,this._currentOffset.y,this._currentOffset.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<32;e++){let t=(e+1)%32;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(64+e*2,32+e,32+t),this._attrIndex.setXY(128+e*2,64+e,64+t)}this._attrIndex.needsUpdate=!0}},jr=new i,Mr=class extends b{constructor(e){if(super(),this.matrixAutoUpdate=!1,this.collider=e,this.collider.shape instanceof Dr)this._geometry=new Ar(this.collider.shape);else if(this.collider.shape instanceof Sr)this._geometry=new Or(this.collider.shape);else if(this.collider.shape instanceof Tr)this._geometry=new kr(this.collider.shape);else throw Error(`VRMSpringBoneColliderHelper: Unknown collider shape type detected`);let t=new ce({color:16711935,depthTest:!1,depthWrite:!1});this._line=new p(this._geometry,t),this.add(this._line)}dispose(){this._geometry.dispose()}updateMatrixWorld(e){this.collider.updateWorldMatrix(!0,!1),this.matrix.copy(this.collider.matrixWorld);let t=this.matrix.elements;this._geometry.worldScale=jr.set(t[0],t[1],t[2]).length(),this._geometry.update(),super.updateMatrixWorld(e)}},Nr=class extends y{constructor(e){super(),this.worldScale=1,this._currentRadius=0,this._currentTail=new i,this._springBone=e,this._attrPos=new m(new Float32Array(294),3),this.setAttribute(`position`,this._attrPos),this._attrIndex=new m(new Uint16Array(194),1),this.setIndex(this._attrIndex),this._buildIndex(),this.update()}update(){let e=!1,t=this._springBone.settings.hitRadius/this.worldScale;this._currentRadius!==t&&(this._currentRadius=t,e=!0),this._currentTail.equals(this._springBone.initialLocalChildPosition)||(this._currentTail.copy(this._springBone.initialLocalChildPosition),e=!0),e&&this._buildPosition()}_buildPosition(){for(let e=0;e<32;e++){let t=e/16*Math.PI;this._attrPos.setXYZ(e,Math.cos(t),Math.sin(t),0),this._attrPos.setXYZ(32+e,0,Math.cos(t),Math.sin(t)),this._attrPos.setXYZ(64+e,Math.sin(t),0,Math.cos(t))}this.scale(this._currentRadius,this._currentRadius,this._currentRadius),this.translate(this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.setXYZ(96,0,0,0),this._attrPos.setXYZ(97,this._currentTail.x,this._currentTail.y,this._currentTail.z),this._attrPos.needsUpdate=!0}_buildIndex(){for(let e=0;e<32;e++){let t=(e+1)%32;this._attrIndex.setXY(e*2,e,t),this._attrIndex.setXY(64+e*2,32+e,32+t),this._attrIndex.setXY(128+e*2,64+e,64+t)}this._attrIndex.setXY(192,96,97),this._attrIndex.needsUpdate=!0}},Pr=new i,Fr=class extends b{constructor(e){super(),this.matrixAutoUpdate=!1,this.springBone=e,this._geometry=new Nr(this.springBone);let t=new ce({color:16776960,depthTest:!1,depthWrite:!1});this._line=new p(this._geometry,t),this.add(this._line)}dispose(){this._geometry.dispose()}updateMatrixWorld(e){this.springBone.bone.updateWorldMatrix(!0,!1),this.matrix.copy(this.springBone.bone.matrixWorld);let t=this.matrix.elements;this._geometry.worldScale=Pr.set(t[0],t[1],t[2]).length(),this._geometry.update(),super.updateMatrixWorld(e)}},Ir=class extends n{constructor(e){super(),this.colliderMatrix=new C,this.shape=e}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),Lr(this.colliderMatrix,this.matrixWorld,this.shape.offset)}};function Lr(e,t,n){let r=t.elements;e.copy(t),n&&(e.elements[12]=r[0]*n.x+r[4]*n.y+r[8]*n.z+r[12],e.elements[13]=r[1]*n.x+r[5]*n.y+r[9]*n.z+r[13],e.elements[14]=r[2]*n.x+r[6]*n.y+r[10]*n.z+r[14])}var Rr=new C;function zr(e){return e.invert?e.invert():e.getInverse(Rr.copy(e)),e}var Br=class{constructor(e){this._inverseCache=new C,this._shouldUpdateInverse=!0,this.matrix=e;let t={set:(e,t,n)=>(this._shouldUpdateInverse=!0,e[t]=n,!0)};this._originalElements=e.elements,e.elements=new Proxy(e.elements,t)}get inverse(){return this._shouldUpdateInverse&&=(zr(this._inverseCache.copy(this.matrix)),!1),this._inverseCache}revert(){this.matrix.elements=this._originalElements}},Vr=new C,J=new i,Y=new i,Hr=new i,Ur=new i,Wr=new C,Gr=class{constructor(e,t,n={},r=[]){this._currentTail=new i,this._prevTail=new i,this._boneAxis=new i,this._worldSpaceBoneLength=0,this._center=null,this._initialLocalMatrix=new C,this._initialLocalRotation=new s,this._initialLocalChildPosition=new i,this.bone=e,this.bone.matrixAutoUpdate=!1,this.child=t,this.settings={hitRadius:n.hitRadius??0,stiffness:n.stiffness??1,gravityPower:n.gravityPower??0,gravityDir:n.gravityDir?.clone()??new i(0,-1,0),dragForce:n.dragForce??.4},this.colliderGroups=r}get dependencies(){let e=new Set,t=this.bone.parent;t&&e.add(t);for(let t=0;t<this.colliderGroups.length;t++)for(let n=0;n<this.colliderGroups[t].colliders.length;n++)e.add(this.colliderGroups[t].colliders[n]);return e}get center(){return this._center}set center(e){this._center?.userData.inverseCacheProxy&&(this._center.userData.inverseCacheProxy.revert(),delete this._center.userData.inverseCacheProxy),this._center=e,this._center&&(this._center.userData.inverseCacheProxy||(this._center.userData.inverseCacheProxy=new Br(this._center.matrixWorld)))}get initialLocalChildPosition(){return this._initialLocalChildPosition}get _parentMatrixWorld(){return this.bone.parent?this.bone.parent.matrixWorld:Vr}setInitState(){this._initialLocalMatrix.copy(this.bone.matrix),this._initialLocalRotation.copy(this.bone.quaternion),this.child?this._initialLocalChildPosition.copy(this.child.position):this._initialLocalChildPosition.copy(this.bone.position).normalize().multiplyScalar(.07);let e=this._getMatrixWorldToCenter();this.bone.localToWorld(this._currentTail.copy(this._initialLocalChildPosition)).applyMatrix4(e),this._prevTail.copy(this._currentTail),this._boneAxis.copy(this._initialLocalChildPosition).normalize()}reset(){this.bone.quaternion.copy(this._initialLocalRotation),this.bone.updateMatrix(),this.bone.matrixWorld.multiplyMatrices(this._parentMatrixWorld,this.bone.matrix);let e=this._getMatrixWorldToCenter();this.bone.localToWorld(this._currentTail.copy(this._initialLocalChildPosition)).applyMatrix4(e),this._prevTail.copy(this._currentTail)}update(e){if(e<=0)return;this._calcWorldSpaceBoneLength();let t=Y.copy(this._boneAxis).transformDirection(this._initialLocalMatrix).transformDirection(this._parentMatrixWorld);Ur.copy(this._currentTail).add(J.subVectors(this._currentTail,this._prevTail).multiplyScalar(1-this.settings.dragForce)).applyMatrix4(this._getMatrixCenterToWorld()).addScaledVector(t,this.settings.stiffness*e).addScaledVector(this.settings.gravityDir,this.settings.gravityPower*e),Hr.setFromMatrixPosition(this.bone.matrixWorld),Ur.sub(Hr).normalize().multiplyScalar(this._worldSpaceBoneLength).add(Hr),this._collision(Ur),this._prevTail.copy(this._currentTail),this._currentTail.copy(Ur).applyMatrix4(this._getMatrixWorldToCenter());let n=Wr.multiplyMatrices(this._parentMatrixWorld,this._initialLocalMatrix).invert();this.bone.quaternion.setFromUnitVectors(this._boneAxis,J.copy(Ur).applyMatrix4(n).normalize()).premultiply(this._initialLocalRotation),this.bone.updateMatrix(),this.bone.matrixWorld.multiplyMatrices(this._parentMatrixWorld,this.bone.matrix)}_collision(e){for(let t=0;t<this.colliderGroups.length;t++)for(let n=0;n<this.colliderGroups[t].colliders.length;n++){let r=this.colliderGroups[t].colliders[n],i=r.shape.calculateCollision(r.colliderMatrix,e,this.settings.hitRadius,J);if(i<0){e.addScaledVector(J,-i),e.sub(Hr);let t=e.length();e.multiplyScalar(this._worldSpaceBoneLength/t).add(Hr)}}}_calcWorldSpaceBoneLength(){J.setFromMatrixPosition(this.bone.matrixWorld),this.child?Y.setFromMatrixPosition(this.child.matrixWorld):(Y.copy(this._initialLocalChildPosition),Y.applyMatrix4(this.bone.matrixWorld)),this._worldSpaceBoneLength=J.sub(Y).length()}_getMatrixCenterToWorld(){return this._center?this._center.matrixWorld:Vr}_getMatrixWorldToCenter(){return this._center?this._center.userData.inverseCacheProxy.inverse:Vr}};function Kr(e,t){let n=[],r=e;for(;r!==null;)n.unshift(r),r=r.parent;n.forEach(e=>{t(e)})}function qr(e,t){e.children.forEach(e=>{t(e)||qr(e,t)})}function Jr(e){let t=new Map;for(let n of e){let r=n;do{let n=(t.get(r)??0)+1;if(n===e.size)return r;t.set(r,n),r=r.parent}while(r!==null)}return null}var Yr=class{constructor(){this._joints=new Set,this._sortedJoints=[],this._hasWarnedCircularDependency=!1,this._ancestors=[],this._objectSpringBonesMap=new Map,this._isSortedJointsDirty=!1,this._relevantChildrenUpdated=this._relevantChildrenUpdated.bind(this)}get joints(){return this._joints}get springBones(){return this._joints}get colliderGroups(){let e=new Set;return this._joints.forEach(t=>{t.colliderGroups.forEach(t=>{e.add(t)})}),Array.from(e)}get colliders(){let e=new Set;return this.colliderGroups.forEach(t=>{t.colliders.forEach(t=>{e.add(t)})}),Array.from(e)}addJoint(e){this._joints.add(e);let t=this._objectSpringBonesMap.get(e.bone);t??(t=new Set,this._objectSpringBonesMap.set(e.bone,t)),t.add(e),this._isSortedJointsDirty=!0}addSpringBone(e){this.addJoint(e)}deleteJoint(e){this._joints.delete(e),this._objectSpringBonesMap.get(e.bone).delete(e),this._isSortedJointsDirty=!0}deleteSpringBone(e){this.deleteJoint(e)}setInitState(){this._sortJoints();for(let e=0;e<this._sortedJoints.length;e++){let t=this._sortedJoints[e];t.bone.updateMatrix(),t.bone.updateWorldMatrix(!1,!1),t.setInitState()}}reset(){this._sortJoints();for(let e=0;e<this._sortedJoints.length;e++){let t=this._sortedJoints[e];t.bone.updateMatrix(),t.bone.updateWorldMatrix(!1,!1),t.reset()}}update(e){this._sortJoints();for(let e=0;e<this._ancestors.length;e++)this._ancestors[e].updateWorldMatrix(e===0,!1);for(let t=0;t<this._sortedJoints.length;t++){let n=this._sortedJoints[t];n.bone.updateMatrix(),n.bone.updateWorldMatrix(!1,!1),n.update(e),qr(n.bone,this._relevantChildrenUpdated)}}_sortJoints(){if(!this._isSortedJointsDirty)return;let e=[],t=new Set,n=new Set,r=new Set;for(let i of this._joints)this._insertJointSort(i,t,n,e,r);this._sortedJoints=e;let i=Jr(r);this._ancestors=[],i&&(this._ancestors.push(i),qr(i,e=>(this._objectSpringBonesMap.get(e)?.size??0)>0||(this._ancestors.push(e),!1))),this._isSortedJointsDirty=!1}_insertJointSort(e,t,n,r,i){if(n.has(e))return;if(t.has(e)){this._hasWarnedCircularDependency||=!0;return}t.add(e);let a=e.dependencies;for(let e of a){let a=!1,o=null;Kr(e,e=>{let s=this._objectSpringBonesMap.get(e);if(s)for(let e of s)a=!0,this._insertJointSort(e,t,n,r,i);else a||(o=e)}),o&&i.add(o)}r.push(e),n.add(e)}_relevantChildrenUpdated(e){return(this._objectSpringBonesMap.get(e)?.size??0)>0||(e.updateWorldMatrix(!1,!1),!1)}},Xr=`VRMC_springBone_extended_collider`,Zr=new Set([`1.0`,`1.0-beta`]),Qr=new Set([`1.0`]),$r=class e{get name(){return e.EXTENSION_NAME}constructor(e,t){this.parser=e,this.jointHelperRoot=t?.jointHelperRoot,this.colliderHelperRoot=t?.colliderHelperRoot,this.useExtendedColliders=t?.useExtendedColliders??!0}afterRoot(e){return yr(this,null,function*(){e.userData.vrmSpringBoneManager=yield this._import(e)})}_import(e){return yr(this,null,function*(){return(yield this._v1Import(e))??(yield this._v0Import(e))??null})}_v1Import(t){return yr(this,null,function*(){var n;let r=t.parser.json;if(r.extensionsUsed?.indexOf(e.EXTENSION_NAME)===-1)return null;let a=new Yr,o=yield t.parser.getDependencies(`node`),s=r.extensions?.[e.EXTENSION_NAME];if(!s)return null;let c=s.specVersion;if(!Zr.has(c))return null;let l=s.colliders?.map((e,t)=>{let n=o[e.node];if(n==null)return null;let r=e.shape,a=e.extensions?.[Xr];if(this.useExtendedColliders&&a!=null){let e=a.specVersion;if(Qr.has(e)){let e=a.shape;if(e.sphere)return this._importSphereCollider(n,{offset:new i().fromArray(e.sphere.offset??[0,0,0]),radius:e.sphere.radius??0,inside:e.sphere.inside??!1});if(e.capsule)return this._importCapsuleCollider(n,{offset:new i().fromArray(e.capsule.offset??[0,0,0]),radius:e.capsule.radius??0,tail:new i().fromArray(e.capsule.tail??[0,0,0]),inside:e.capsule.inside??!1});if(e.plane)return this._importPlaneCollider(n,{offset:new i().fromArray(e.plane.offset??[0,0,0]),normal:new i().fromArray(e.plane.normal??[0,0,1])})}}if(r.sphere)return this._importSphereCollider(n,{offset:new i().fromArray(r.sphere.offset??[0,0,0]),radius:r.sphere.radius??0,inside:!1});if(r.capsule)return this._importCapsuleCollider(n,{offset:new i().fromArray(r.capsule.offset??[0,0,0]),radius:r.capsule.radius??0,tail:new i().fromArray(r.capsule.tail??[0,0,0]),inside:!1})}),u=s.colliderGroups?.map((e,t)=>({colliders:(e.colliders??[]).map(e=>l?.[e]??null).filter(e=>e!=null),name:e.name}));return(n=s.springs)==null||n.forEach((e,t)=>{let n=e.joints,r=e.colliderGroups?.map(e=>u?.[e]??null).filter(e=>e!=null),s=e.center==null?void 0:o[e.center],c;n.forEach(e=>{if(c){let t=c.node,n=o[t],l=e.node,u=o[l],d={hitRadius:c.hitRadius,dragForce:c.dragForce,gravityPower:c.gravityPower,stiffness:c.stiffness,gravityDir:c.gravityDir==null?void 0:new i().fromArray(c.gravityDir)},f=this._importJoint(n,u,d,r);s&&(f.center=s),a.addJoint(f)}c=e})}),a.setInitState(),a})}_v0Import(e){return yr(this,null,function*(){let t=e.parser.json;if(t.extensionsUsed?.indexOf(`VRM`)===-1)return null;let n=t.extensions?.VRM?.secondaryAnimation;if(!n)return null;let r=n?.boneGroups;if(!r)return null;let a=new Yr,o=yield e.parser.getDependencies(`node`),s=n.colliderGroups?.map((e,t)=>{let n=o[e.node];return n==null?null:{colliders:(e.colliders??[]).map((e,t)=>{let r=new i(0,0,0);return e.offset&&r.set(e.offset.x??0,e.offset.y??0,e.offset.z?-e.offset.z:0),this._importSphereCollider(n,{offset:r,radius:e.radius??0,inside:!1})})}});return r?.forEach((e,t)=>{let n=e.bones;n&&n.forEach(t=>{let n=o[t];if(n==null)return;let r=new i;e.gravityDir?r.set(e.gravityDir.x??0,e.gravityDir.y??0,e.gravityDir.z??0):r.set(0,-1,0);let c=e.center==null?void 0:o[e.center],l={hitRadius:e.hitRadius,dragForce:e.dragForce,gravityPower:e.gravityPower,stiffness:e.stiffiness,gravityDir:r},u=e.colliderGroups?.map(e=>s?.[e]??null).filter(e=>e!=null);n.traverse(e=>{let t=e.children[0]??null,n=this._importJoint(e,t,l,u);c&&(n.center=c),a.addJoint(n)})})}),e.scene.updateMatrixWorld(),a.setInitState(),a})}_importJoint(e,t,n,r){let i=new Gr(e,t,n,r);if(this.jointHelperRoot){let e=new Fr(i);this.jointHelperRoot.add(e),e.renderOrder=this.jointHelperRoot.renderOrder}return i}_importSphereCollider(e,t){let n=new Ir(new Dr(t));if(e.add(n),this.colliderHelperRoot){let e=new Mr(n);this.colliderHelperRoot.add(e),e.renderOrder=this.colliderHelperRoot.renderOrder}return n}_importCapsuleCollider(e,t){let n=new Ir(new Sr(t));if(e.add(n),this.colliderHelperRoot){let e=new Mr(n);this.colliderHelperRoot.add(e),e.renderOrder=this.colliderHelperRoot.renderOrder}return n}_importPlaneCollider(e,t){let n=new Ir(new Tr(t));if(e.add(n),this.colliderHelperRoot){let e=new Mr(n);this.colliderHelperRoot.add(e),e.renderOrder=this.colliderHelperRoot.renderOrder}return n}};$r.EXTENSION_NAME=`VRMC_springBone`;var ei=$r,ti=class{get name(){return`VRMLoaderPlugin`}constructor(e,t){this.parser=e;let n=t?.helperRoot,r=t?.autoUpdateHumanBones;this.expressionPlugin=t?.expressionPlugin??new ut(e),this.firstPersonPlugin=t?.firstPersonPlugin??new mt(e),this.humanoidPlugin=t?.humanoidPlugin??new jt(e,{helperRoot:n,autoUpdateHumanBones:r}),this.lookAtPlugin=t?.lookAtPlugin??new un(e,{helperRoot:n}),this.metaPlugin=t?.metaPlugin??new pn(e),this.mtoonMaterialPlugin=t?.mtoonMaterialPlugin??new Mn(e),this.materialsHDREmissiveMultiplierPlugin=t?.materialsHDREmissiveMultiplierPlugin??new Fn(e),this.materialsV0CompatPlugin=t?.materialsV0CompatPlugin??new Gn(e),this.springBonePlugin=t?.springBonePlugin??new ei(e,{colliderHelperRoot:n,jointHelperRoot:n}),this.nodeConstraintPlugin=t?.nodeConstraintPlugin??new vr(e,{helperRoot:n})}beforeRoot(){return Ge(this,null,function*(){yield this.materialsV0CompatPlugin.beforeRoot(),yield this.mtoonMaterialPlugin.beforeRoot()})}loadMesh(e){return Ge(this,null,function*(){return yield this.mtoonMaterialPlugin.loadMesh(e)})}getMaterialType(e){return this.mtoonMaterialPlugin.getMaterialType(e)??null}extendMaterialParams(e,t){return Ge(this,null,function*(){yield this.materialsHDREmissiveMultiplierPlugin.extendMaterialParams(e,t),yield this.mtoonMaterialPlugin.extendMaterialParams(e,t)})}afterRoot(e){return Ge(this,null,function*(){yield this.metaPlugin.afterRoot(e),yield this.humanoidPlugin.afterRoot(e),yield this.expressionPlugin.afterRoot(e),yield this.lookAtPlugin.afterRoot(e),yield this.firstPersonPlugin.afterRoot(e),yield this.springBonePlugin.afterRoot(e),yield this.nodeConstraintPlugin.afterRoot(e),yield this.mtoonMaterialPlugin.afterRoot(e);let t=e.userData.vrmMeta,n=e.userData.vrmHumanoid;if(t&&n){let r=new hn({scene:e.scene,expressionManager:e.userData.vrmExpressionManager,firstPerson:e.userData.vrmFirstPerson,humanoid:n,lookAt:e.userData.vrmLookAt,meta:t,materials:e.userData.vrmMToonMaterials,springBoneManager:e.userData.vrmSpringBoneManager,nodeConstraintManager:e.userData.vrmNodeConstraintManager});e.userData.vrm=r}})}};function ni(e){let t=new Set;return e.traverse(e=>{if(!e.isMesh)return;let n=e;t.add(n)}),t}function ri(e,t,n){if(t.size===1){let n=t.values().next().value;if(n.weight===1)return e[n.index]}let r=new Float32Array(e[0].count*3),i=0;if(n)i=1;else for(let e of t)i+=e.weight;for(let n of t){let t=e[n.index],a=n.weight/i;for(let e=0;e<t.count;e++)r[e*3+0]+=t.getX(e)*a,r[e*3+1]+=t.getY(e)*a,r[e*3+2]+=t.getZ(e)*a}return new m(r,3)}function ii(e){let t=ni(e.scene),n=new Map,r=e.expressionManager?.expressionMap;if(r!=null)for(let[e,t]of Object.entries(r)){let r=new Set;for(let i of t.binds)if(i instanceof it){if(i.weight!==0)for(let t of i.primitives){let r=n.get(t);r??(r=new Map,n.set(t,r));let a=r.get(e);a??(a=new Set,r.set(e,a)),a.add(i)}r.add(i)}for(let e of r)t.deleteBind(e)}for(let e of t){let t=n.get(e);if(t==null)continue;let i=e.geometry.morphAttributes;e.geometry.morphAttributes={};let a=e.geometry.clone();e.geometry=a;let o=a.morphTargetsRelative,s=i.position!=null,c=i.normal!=null,l={},u={},d=[];if(s||c){s&&(l.position=[]),c&&(l.normal=[]);let n=0;for(let[a,f]of t)s&&(l.position[n]=ri(i.position,f,o)),c&&(l.normal[n]=ri(i.normal,f,o)),r?.[a].addBind(new it({index:n,weight:1,primitives:[e]})),u[a]=n,d.push(0),n++}a.morphAttributes=l,e.morphTargetDictionary=u,e.morphTargetInfluences=d}}function ai(e,t,n){if(e.getComponent)return e.getComponent(t,n);{let r=e.array[t*e.itemSize+n];return e.normalized&&(r=l.denormalize(r,e.array)),r}}function oi(e,t,n,r){e.setComponent?e.setComponent(t,n,r):(e.normalized&&(r=l.normalize(r,e.array)),e.array[t*e.itemSize+n]=r)}function si(e){let t=ci(e),n=new Set;for(let e of t)n.has(e.geometry)&&(e.geometry=hi(e.geometry)),n.add(e.geometry);let r=new Map;for(let e of n){let t=e.getAttribute(`skinIndex`),n=r.get(t)??new Map;r.set(t,n);let i=e.getAttribute(`skinWeight`),a=li(t,i);n.set(i,a)}let i=new Map;for(let e of t){let t=ui(e,r);i.set(e,t)}let a=[];for(let[e,t]of i){let n=!1;for(let r of a)if(di(t,r.boneInverseMap)){n=!0,r.meshes.add(e);for(let[e,n]of t)r.boneInverseMap.set(e,n);break}n||a.push({boneInverseMap:t,meshes:new Set([e])})}let o=new Map,s=new mi,c=new mi,l=new mi;for(let e of a){let{boneInverseMap:t,meshes:n}=e,r=Array.from(t.keys()),i=Array.from(t.values()),a=new te(r,i),u=c.getOrCreate(a);for(let e of n){let t=e.geometry.getAttribute(`skinIndex`),n=s.getOrCreate(t),i=e.skeleton.bones,a=`${n};${u};${i.map(e=>l.getOrCreate(e)).join(`,`)}`,c=o.get(a);c??(c=t.clone(),fi(c,i,r),o.set(a,c)),e.geometry.setAttribute(`skinIndex`,c)}for(let e of n)e.bind(a,new C)}}function ci(e){let t=new Set;return e.traverse(e=>{if(!e.isSkinnedMesh)return;let n=e;t.add(n)}),t}function li(e,t){let n=new Set;for(let r=0;r<e.count;r++)for(let i=0;i<e.itemSize;i++){let a=ai(e,r,i);ai(t,r,i)!==0&&n.add(a)}return n}function ui(e,t){let n=new Map,r=e.skeleton,i=e.geometry,a=i.getAttribute(`skinIndex`),o=i.getAttribute(`skinWeight`),s=t.get(a)?.get(o);if(!s)throw Error(`Unreachable. attributeUsedIndexSetMap does not know the skin index attribute or the skin weight attribute.`);for(let e of s)n.set(r.bones[e],r.boneInverses[e]);return n}function di(e,t){for(let[n,r]of e.entries()){let e=t.get(n);if(e!=null&&!pi(r,e))return!1}return!0}function fi(e,t,n){let r=new Map;for(let e of t)r.set(e,r.size);let i=new Map;for(let[e,t]of n.entries()){let n=r.get(t);i.set(n,e)}for(let t=0;t<e.count;t++)for(let n=0;n<e.itemSize;n++){let r=ai(e,t,n),a=i.get(r);oi(e,t,n,a)}e.needsUpdate=!0}function pi(e,t,n){if(n||=1e-4,e.elements.length!=t.elements.length)return!1;for(let r=0,i=e.elements.length;r<i;r++)if(Math.abs(e.elements[r]-t.elements[r])>n)return!1;return!0}var mi=class{constructor(){this._objectIndexMap=new Map,this._index=0}get(e){return this._objectIndexMap.get(e)}getOrCreate(e){let t=this._objectIndexMap.get(e);return t??(t=this._index,this._objectIndexMap.set(e,t),this._index++),t}};function hi(e){let t=new y;t.name=e.name,t.setIndex(e.index);for(let[n,r]of Object.entries(e.attributes))t.setAttribute(n,r);for(let[n,r]of Object.entries(e.morphAttributes)){let e=n;t.morphAttributes[e]=r.concat()}t.morphTargetsRelative=e.morphTargetsRelative,t.groups=[];for(let n of e.groups)t.addGroup(n.start,n.count,n.materialIndex);return t.boundingSphere=e.boundingSphere?.clone()??null,t.boundingBox=e.boundingBox?.clone()??null,t.drawRange.start=e.drawRange.start,t.drawRange.count=e.drawRange.count,t.userData=e.userData,t}function gi(e){if(Object.values(e).forEach(e=>{e?.isTexture&&e.dispose()}),e.isShaderMaterial){let t=e.uniforms;t&&Object.values(t).forEach(e=>{let t=e.value;t?.isTexture&&t.dispose()})}e.dispose()}function _i(e){let t=e.geometry;t&&t.dispose();let n=e.skeleton;n&&n.dispose();let r=e.material;r&&(Array.isArray(r)?r.forEach(e=>gi(e)):r&&gi(r))}function vi(e){e.traverse(_i)}function yi(e,t){let n=t?.experimentalSameBoneCounts??!1,r=[];e.traverse(e=>{e.type===`SkinnedMesh`&&r.push(e)});let i=new Map,a=0;for(let e of r){let t=e.geometry.getAttribute(`skinIndex`);if(i.has(t))continue;let n=new Map,r=new Map;for(let e=0;e<t.count;e++)for(let i=0;i<t.itemSize;i++){let a=ai(t,e,i),o=n.get(a);o??(o=n.size,n.set(a,o),r.set(o,a)),oi(t,e,i,o)}t.needsUpdate=!0,i.set(t,r),a=Math.max(a,n.size)}for(let e of r){let t=e.geometry.getAttribute(`skinIndex`),r=i.get(t),o=[],s=[],c=n?a:r.size;for(let t=0;t<c;t++){let n=r.get(t)??0;o.push(e.skeleton.bones[n]),s.push(e.skeleton.boneInverses[n])}let l=new te(o,s);e.bind(l,new C)}}function bi(e,t){let n=e.position.count,r=Array(n),i=0,a=t.array;for(let e=0;e<a.length;e++){let t=a[e];r[t]||(r[t]=!0,i++)}return{isVertexUsed:r,vertexCount:n,verticesUsed:i}}function xi(e){let t=[],n=[],r=0;for(let i=0;i<e.length;i++)if(e[i]){let e=r++;t[i]=e,n[e]=i}return{originalIndexNewIndexMap:t,newIndexOriginalIndexMap:n}}function Si(e,t){t.name=e.name,t.morphTargetsRelative=e.morphTargetsRelative,e.groups.forEach(e=>{t.addGroup(e.start,e.count,e.materialIndex)}),t.boundingBox=e.boundingBox?.clone()??null,t.boundingSphere=e.boundingSphere?.clone()??null,t.setDrawRange(e.drawRange.start,e.drawRange.count),t.userData=e.userData}function Ci(e,t,n){let r=t.array,i=new r.constructor(r.length);for(let e=0;e<r.length;e++){let t=r[e];i[e]=n[t]}e.setIndex(new m(i,t.itemSize,t.normalized))}function wi(e,t,n){let r=e.constructor,i=new r(t.length*n),a=!0;for(let r=0;r<t.length;r++){let o=t[r]*n,s=r*n;for(let t=0;t<n;t++){let n=e[o+t];i[s+t]=n,a&&=n===0}}return[i,a]}function Ti(e){let t=new Map,n=[];for(let[r,i]of Object.entries(e))if(i.isInterleavedBufferAttribute){let e=i,n=e.data,a=t.get(n)??[];t.set(n,a),a.push([r,e])}else{let e=i;n.push([r,e])}return[t,n]}function Ei(e,t,n){let[r,i]=Ti(t);for(let[t,i]of r){let r=t.array,{stride:a}=t,[o,s]=wi(r,n,a),c=new _(o,a);c.setUsage(t.usage);for(let[t,n]of i){let{itemSize:r,offset:i,normalized:a}=n,o=new d(c,r,i,a);e.setAttribute(t,o)}}for(let[t,r]of i){let i=r.array,{itemSize:a,normalized:o}=r,[s,c]=wi(i,n,a);e.setAttribute(t,new m(s,a,o))}}function Di(e){let t=new Map,n=[];for(let[r,i]of Object.entries(e)){let e=r;for(let r=0;r<i.length;r++){let a=i[r];if(a.isInterleavedBufferAttribute){let n=a,i=n.data,o=t.get(i)??[];t.set(i,o),o.push([e,r,n])}else{let t=a;n.push([e,r,t])}}}return[t,n]}function Oi(e,t,n){let r=!0,[i,a]=Di(t),o={};for(let[e,t]of i){let i=e.array,{stride:a}=e,[s,c]=wi(i,n,a);r&&=c;let l=new _(s,a);l.setUsage(e.usage);for(let[e,n,r]of t){let{itemSize:t,offset:i,normalized:a}=r,s=new d(l,t,i,a);o[e]??(o[e]=[]),o[e][n]=s}}for(let[e,t,i]of a){let a=i,s=a.array,{itemSize:c,normalized:l}=a,[u,d]=wi(s,n,c);r&&=d,o[e]??(o[e]=[]),o[e][t]=new m(u,c,l)}e.morphAttributes=r?{}:o}function ki(e){let t=new Map;e.traverse(e=>{if(!e.isMesh)return;let n=e,r=n.geometry,i=r.index;if(i==null)return;let a=t.get(r);if(a!=null){n.geometry=a;return}let{isVertexUsed:o,vertexCount:s,verticesUsed:c}=bi(r.attributes,i);if(c===s)return;let{originalIndexNewIndexMap:l,newIndexOriginalIndexMap:u}=xi(o),d=new y;Si(r,d),t.set(r,d),Ci(d,i,l),Ei(d,r.attributes,u),Oi(d,r.morphAttributes,u),n.geometry=d}),Array.from(t.keys()).forEach(e=>{e.dispose()})}function Ai(e){e.meta?.metaVersion===`0`&&(e.scene.rotation.y=Math.PI)}var X=class{constructor(){}};X.combineMorphs=ii,X.combineSkeletons=si,X.deepDispose=vi,X.removeUnnecessaryJoints=yi,X.removeUnnecessaryVertices=ki,X.rotateVRM0=Ai;var ji=65536,Mi=256,Ni=1024,Pi=65536,Fi=8192,Ii=1<<22,Li=256;function Ri(e){let t=0;for(let n=e;n;n=n.parent)t++;return t}function zi(e){let t=0;for(let n of e.nodeConstraintManager?.constraints??[])for(let e of[...n.dependencies,n.source,n.destination]){let n=Ri(e);if(n>Mi)throw Error(`VRM node constraint on a node ${n} deep (at most ${Mi})`);t+=n}if(t>ji)throw Error(`VRM node constraints visit ${t} nodes per update (at most ${ji})`)}function Bi(e){let t=0,n=0;for(let r of e){if(!Array.isArray(r.joints)||!Array.isArray(r.colliders))throw Error(`a spring chain needs joints and colliders lists`);let e=Math.max(0,r.joints.length-+(r.tip===void 0));t+=e,n+=e*r.colliders.length}if(!(t<=Ni&&n<=Pi))throw Error(`a skin's springs make ${t} joints and ${n} collider tests per update (at most ${Ni} and ${Pi})`)}var Vi=class extends ei{async afterRoot(e){let t=await e.parser.getDependencies(`node`),n=Ui(Hi(e.parser.json,t));if(n)throw Error(`a skin's springs exceed ${n}`);return super.afterRoot(e)}},Z=e=>Array.isArray(e)?e:[],Q=(e,t)=>typeof e==`object`&&e?e[t]:void 0;function Hi(e,t){let n=new Map,r=e=>{let t=e;for(;t.parent;)t=t.parent;if(!n.has(t)){n.set(t,{parent:null,children:[],joints:0});for(let e=[t];e.length;){let t=e.pop(),r=n.get(t);for(let i of t.children){let t={parent:r,children:[],joints:0};n.set(i,t),r.children.push(t),e.push(i)}}}return n.get(e)},i=e=>{if(!e)return null;let t=r(e),n={parent:t,children:[],joints:0};return t.children.push(n),n},a=(e,t)=>{let n=Z(e).map(e=>t[e]).filter(e=>!!e);return{groups:n,tests:n.reduce((e,t)=>e+1+t.length,0)}},o={joints:0,tests:0,updates:0,sort:0,depth:0},s=[],c=(e,t)=>(e.joints++,s.push({bone:e,groups:t.groups}),o.joints++,o.tests+=t.tests,o.joints>Ni||o.tests>Pi),l=e=>t[e],u=Q(Q(e,`extensions`),`VRMC_springBone`),d=Z(Q(u,`colliders`)).map(e=>i(l(Q(e,`node`)))),f=Z(Q(u,`colliderGroups`)).map(e=>Z(Q(e,`colliders`)).map(e=>d[e]).filter(e=>!!e));for(let e of Z(Q(u,`springs`))){let t=a(Q(e,`colliderGroups`),f),n=Z(Q(e,`joints`));for(let e=0;e+1<n.length;e++){let i=l(Q(n[e],`node`));if(i&&c(r(i),t))return o}}let p=Q(Q(Q(e,`extensions`),`VRM`),`secondaryAnimation`),m=Z(Q(p,`colliderGroups`)).map(e=>{let t=l(Q(e,`node`));return t?Z(Q(e,`colliders`)).map(()=>i(t)):null});for(let e of Z(Q(p,`boneGroups`))){let t=a(Q(e,`colliderGroups`),m);for(let n of Z(Q(e,`bones`))){let e=l(n);for(let n=e?[r(e)]:[];n.length;){let e=n.pop();if(c(e,t))return o;for(let t of e.children)n.push(t)}}}let h=new Map;o.sort=s.length;for(let{bone:e,groups:t}of s){if(!h.has(e)){let t=0;for(let n=[...e.children];n.length;){let e=n.pop();if(!e.joints){t++;for(let t of e.children)n.push(t)}}h.set(e,t)}if(o.updates+=h.get(e),o.updates>Fi)return o;let n=new Set(e.parent?[e.parent]:[]);for(let e of new Set(t))for(let t of e)n.add(t);for(let e of n){let t=0;for(let n=e;n;n=n.parent)t++,o.sort+=1+n.joints;if(o.depth=Math.max(o.depth,t),t>Li||o.sort>Ii)return o}}return o}function Ui(e){return e.joints>Ni?`${Ni} joints`:e.tests>Pi?`${Pi} collider tests a step`:e.updates>Fi?`${Fi} updates a step`:e.depth>Li?`a depth of ${Li}`:e.sort>Ii?`${Ii} sort steps`:null}function Wi(e){return{root:`hips`,get autoUpdate(){return e.autoUpdateHumanBones},set autoUpdate(t){e.autoUpdateHumanBones=t},node:t=>e.getNormalizedBoneNode(t),rawNode:t=>e.getRawBoneNode(t),ancestors:e=>{let t=[];for(let n=bt[e];n;n=bt[n])t.push(n);return t},update:()=>e.update(),resetRawPose:()=>e.resetRawPose()}}var Gi=new i,Ki=new i,qi=new i,Ji=class{constructor(e){this.normalizedRoot=new n,this.autoUpdate=!0,this.bones=new Map;let t=new Map(Object.entries(e).map(([e,t])=>[t,e])),r=e=>{let n=[];for(let r=e.parent;r;r=r.parent){let e=t.get(r);e!==void 0&&n.push(e)}return n},a=Object.entries(e).map(([e,t])=>({bone:e,node:t,ancestors:r(t)})).sort((e,t)=>e.ancestors.length-t.ancestors.length),o=new Map;for(let{bone:e,node:t,ancestors:r}of a){t.updateWorldMatrix(!0,!1);let a=new i().setFromMatrixPosition(t.matrixWorld);o.set(e,a);let c=new s;t.parent?.matrixWorld.decompose(qi,c,Ki);let l=new n;l.name=`Normalized_${t.name}`,(r.length?this.bones.get(r[0]).normalized:this.normalizedRoot).add(l),l.position.copy(a),r.length&&l.position.sub(o.get(r[0])),this.bones.set(e,{raw:t,normalized:l,ancestors:r,parentTurn:c,parentTurnInverse:c.clone().invert(),turn:t.quaternion.clone(),position:t.position.clone()})}this.root=a.find(e=>!e.ancestors.length)?.bone??``}node(e){return this.bones.get(e)?.normalized??null}rawNode(e){return this.bones.get(e)?.raw??null}ancestors(e){return[...this.bones.get(e)?.ancestors??[]]}update(){if(!this.autoUpdate)return;for(let e of this.bones.values())e.raw.quaternion.copy(e.normalized.quaternion).multiply(e.parentTurn).premultiply(e.parentTurnInverse).multiply(e.turn);let e=this.bones.get(this.root),t=e?.raw.parent;e&&t&&(e.normalized.getWorldPosition(Gi),t.updateWorldMatrix(!0,!1),e.raw.position.copy(t.worldToLocal(Gi)))}resetRawPose(){for(let e of this.bones.values())e.raw.quaternion.copy(e.turn),e.raw.position.copy(e.position)}},Yi={"+z":0,"+x":Math.PI/2,"-z":Math.PI,"-x":-Math.PI/2},Xi=new i(0,1,0);async function Zi(e,t){let n=new f;t.format===`vrm`&&n.register(e=>new ti(e,{springBonePlugin:new Vi(e)}));let r=await n.parseAsync(e,``);return Qi(r.scene),t.format===`vrm`?$i(r):ea(r,t)}function Qi(e){let t=new Set,n=new Set;e.traverse(e=>{let{material:r,geometry:i}=e;for(let e of r?[r].flat():[])t.add(e);i&&n.add(i)});for(let e of t){let{uniforms:t}=e,n=t?Object.values(t).map(e=>e.value):Object.values(e);for(let e of n){let t=e,n=e;t?.isTexture&&t.colorSpace===`srgb`?(t.colorSpace=``,t.needsUpdate=!0):n?.isColor&&n.convertLinearToSRGB()}}let r=new S;for(let e of n){let t=e.getAttribute(`color`);if(t){for(let e=0;e<t.count;e++)r.setRGB(t.getX(e),t.getY(e),t.getZ(e)).convertLinearToSRGB(),t.setXYZ(e,r.r,r.g,r.b);t.needsUpdate=!0}}}function $i(e){let t=e.userData.vrm;if(!t)throw Error(`not a VRM file (no VRM or VRMC_vrm extension)`);return zi(t),X.removeUnnecessaryVertices(e.scene),X.combineSkeletons(e.scene),t.expressionManager?.update(),{scene:t.scene,rig:Wi(t.humanoid),springBoneManager:t.springBoneManager??null,frame:ye(t.meta?.metaVersion),update(e){t.humanoid.update(),t.lookAt?.update(e),t.nodeConstraintManager?.update();for(let n of t.materials??[])n.update?.(e)},dispose:()=>X.deepDispose(t.scene)}}async function ea(e,t){let n=t=>e.parser.getDependency(`node`,t);e.scene.scale.setScalar(t.scale??1);let r=new b;r.add(e.scene),r.updateMatrixWorld(!0);for(let[e,r]of t.reparent??[])(await n(r)).attach(await n(e));let i={};for(let[e,r]of Object.entries(t.nodes??{}))i[e]=await n(r);let a=new Ji(i);r.add(a.normalizedRoot);let o=t.springs?.length?await na(t.springs,n):null;return t.material?.toon&&ia(e.scene,t.material.toon),{scene:r,rig:a,springBoneManager:o,frame:new s().setFromAxisAngle(Xi,-Yi[t.facing??`+z`]),update:()=>a.update(),dispose:()=>X.deepDispose(r)}}function ta(e){return 1/e.getWorldScale(new i).x}async function na(e,t){Bi(e);let n=new Yr,r=(e,t)=>new i(...e).multiplyScalar(t);for(let a of e){let e=await Promise.all(a.colliders.map(async({node:e,radius:n,offset:i,tail:a})=>{let o=await t(e),s=ta(o),c=new Ir(a?new Sr({radius:n,offset:r(i,s),tail:r(a,s)}):new Dr({radius:n,offset:r(i,s)}));return o.add(c),c})),o={hitRadius:a.hit_radius,stiffness:a.stiffness,dragForce:a.drag,gravityPower:a.gravity,gravityDir:new i(...a.gravity_dir)},s=[{colliders:e}],c=await Promise.all(a.joints.map(t));a.tip!==void 0&&c.push(ra(c[c.length-1],a.tip));for(let e=0;e+1<c.length;e++)n.addJoint(new Gr(c[e],c[e+1],o,s))}return n.setInitState(),n}function ra(e,t){let r=new n;return r.position.copy(e.position).setLength(t*ta(e)),e.add(r),r}function ia(e,t){let n=new Map,r=e=>{let r=e;if(!r.isMeshStandardMaterial)return e;let i=n.get(r);return i||(i=new kn({color:r.color.clone(),map:r.map??void 0,shadeColorFactor:r.color.clone().multiplyScalar(t.shade),shadeMultiplyTexture:r.map??void 0,shadingToonyFactor:t.toony,shadingShiftFactor:t.shift,transparent:r.transparent,alphaTest:r.alphaTest,side:r.side,vertexColors:r.vertexColors}),n.set(r,i),r.dispose()),i};e.traverse(e=>{let t=e;t.isMesh&&(t.material=Array.isArray(t.material)?t.material.map(r):r(t.material))})}var aa=1/120,oa=1/30,sa=.25,ca=.3,la=[0,0,-1],ua=8,da=.25,fa=new i(1,1,1),pa=new C,$=new i,ma=new i,ha=new Int32Array(1);function ga(e,t,n,r){let i=t.geom_type[r],a=e.mjtGeom,[o,s,c]=[6,7,8].map(e=>n.geom_xmat[r*9+e]),[l,u,d]=[0,1,2].map(e=>t.geom_size[r*3+e]),f=n.geom_xpos[r*3+2];if(i===a.mjGEOM_SPHERE.value)return f-l;if(i===a.mjGEOM_CAPSULE.value)return f-Math.abs(c)*u-l;if(i===a.mjGEOM_CYLINDER.value)return f-Math.abs(c)*u-l*Math.sqrt(Math.max(0,1-c*c));if(i===a.mjGEOM_ELLIPSOID.value)return f-Math.hypot(o*l,s*u,c*d);if(i===a.mjGEOM_BOX.value)return f-Math.abs(o)*l-Math.abs(s)*u-Math.abs(c)*d;if(i===a.mjGEOM_PLANE.value||i===a.mjGEOM_HFIELD.value)return 1/0;let[p,m,h,g,_,v]=[0,1,2,3,4,5].map(e=>t.geom_aabb[r*6+e]);return f+o*p+s*m+c*h-Math.abs(o)*g-Math.abs(s)*_-Math.abs(c)*v}function _a(e,t){let n=-1,r=0;for(let i=0;i<e.nbody;i++){let a=e.body(i).name;if(a===t)return i;a.endsWith(`/${t}`)&&(n=i,r+=1)}return r===1?n:-1}var va=class e{constructor(e,t){this.model=e,this.config=t,this.root=new b,this.hiddenBodies=[],this.rig=null,this.nodes=[],this.world=[],this.local=[],this.rootBone=null,this.rootBody=-1,this.robotRoot=-1,this.heightRatio=1,this.mujoco=null,this.poses=new Map,this.hipsOffset=new i,this.rootFrom=new i,this.rootTo=new i,this.rootNow=new i,this.groundFrom=0,this.groundTo=0,this.lastTime=null,this.toon=[],this.springs=null,this.springsBuilt=!1,this.attached=[],this.limbs=[],this.root.name=`Skin`,this.root.add(e.scene);let n=new Set;e.scene.traverse(e=>{e.frustumCulled=!1,e.raycast=()=>{};let t=e;if(t.isMesh){t.castShadow=!0;for(let e of Array.isArray(t.material)?t.material:[t.material])n.add(e)}});for(let e of n){let t=e;t.isMToonMaterial&&this.toon.push({material:t,lit:t.color.clone(),shade:t.shadeColorFactor.clone()})}this.root.updateMatrixWorld(!0);let r=e.rig.node(e.rig.root),a=this.lowestVertex(),o=Number.isFinite(a)?a:0;this.restHeight=r?r.getWorldPosition(new i).y-o:0}static async load(t,n){return new e(await Zi(t,n),n)}bind(e,t){this.springsBuilt||(this.springsBuilt=!0,this.springs=this.buildSprings(e)),this.normalizeToon(t),this.mujoco=e;let n=this.model.rig,r=[];for(let[e,i]of Object.entries(this.config.bones)){let a=n.node(e),o=_a(t,i);!a||o<0||r.push({name:e,bodyId:o,ancestors:n.ancestors(e)})}let a=this.model.frame,o=xe(t,e.mjtJoint.mjJNT_FREE.value);this.rig=Ce(r,o,this.config.rest_offsets??{},a),this.nodes=this.rig.bones.map(e=>n.node(e.name)),this.world=this.rig.bones.map(()=>new s),this.local=this.rig.bones.map(()=>new s),this.poses=new Map(this.rig.bones.map(e=>[e.bodyId,{from:new s,to:new s,now:new s,fromAt:new i,toAt:new i,nowAt:new i}]));let c=this.rig.bones.find(e=>e.name===n.root);if(this.rootBone=c?n.node(c.name):null,this.rootBody=c?.bodyId??-1,this.robotRoot=this.rootBody>=0?t.body_rootid[this.rootBody]:-1,this.hiddenBodies=[],this.config.hide_robot!==!1&&this.rootBody>=0)for(let e=1;e<t.nbody;e++)t.body_rootid[e]===this.robotRoot&&this.hiddenBodies.push(e);let l=this.config.hips_offset??[0,0,0];this.hipsOffset.set(l[0],l[2],-l[1]);let u=this.rootBody>=0?this.hipsOffset.clone().applyQuaternion(o[this.rootBody]).y:0,d=(this.config.root_height??Se(t,o,this.rootBody))+u;this.heightRatio=this.config.height_ratio??(this.restHeight>0&&d>0?this.restHeight/d:1);let f=this.config.offset??[0,0,0];this.root.position.set(f[0],f[2],-f[1]),this.root.quaternion.copy(this.rig.facing).multiply(a);let p=this.config.retarget??`hybrid`,m=p===`position`||p===`hybrid`&&this.config.limbs?.length?this.attach(e,t,p===`hybrid`):[];this.attached=p===`position`?m:[],this.limbs=p===`hybrid`?this.limbsOf(m):[],n.autoUpdate=p!==`position`,this.lastTime=null}limbsOf(e){let t=this.model.rig,n=new Set(this.rig.bones.map(e=>e.name));return(this.config.limbs??[]).flatMap(r=>{let a=e.find(e=>e.name===r[2]);if(!a||!r.every(e=>n.has(e)))return[];let[o,c,l]=r.map(e=>t.node(e)),[u,d,f]=[o,c,l].map(e=>e.getWorldPosition(new i));return[{upper:o,lower:c,end:l,target:a,bend:new i(1,0,0),length:u.distanceTo(d)+d.distanceTo(f),at:new i,turn:new s}]})}attach(e,t,n){let r=this.rig,a=this.model.rig,o=new e.MjData(t);try{t.nkey>0&&e.mj_resetDataKeyframe(t,o,0),e.mj_kinematics(t,o);let c=e=>new i(o.xpos[e*3],o.xpos[e*3+2],-o.xpos[e*3+1]),l=e=>be(o.xquat,e,new s);a.autoUpdate=!0,a.resetRawPose(),Oe(r,l,this.world,this.local);for(let e=0;e<this.nodes.length;e++)this.nodes[e].quaternion.copy(this.local[e]);this.rootBone&&ke(r,this.hipsOffset.clone().applyQuaternion(l(this.rootBody)).add(c(this.rootBody)),this.heightRatio,0,this.rootBone.position),a.update();let u=this.root.position.clone();this.root.position.set(0,0,0),this.root.updateMatrixWorld(!0);let d=this.robotBottom(e,t,o)-this.lowestVertex();this.root.position.y=Number.isFinite(d)?d:0,this.root.updateMatrixWorld(!0);let f=new C,p=r.bones.flatMap(e=>{let t=n?a.node(e.name):a.rawNode(e.name);if(!t)return[];f.compose(c(e.bodyId),l(e.bodyId),fa).invert();let r=f.clone().multiply(t.matrixWorld);return[{name:e.name,bone:t,bodyId:e.bodyId,inBody:r}]});this.root.position.copy(u),this.root.updateMatrixWorld(!0);let m=e=>{let t=0;for(let n=e.parent;n;n=n.parent)t++;return t};return p.sort((e,t)=>m(e.bone)-m(t.bone))}finally{o.delete()}}robotBottom(e,t,n){let r=1/0;for(let i=0;i<t.ngeom;i++)t.body_rootid[t.geom_bodyid[i]]===this.robotRoot&&(t.geom_contype[i]||t.geom_conaffinity[i])&&(r=Math.min(r,ga(e,t,n,i)));return r}lowestVertex(){let e=1/0,t=new i;return this.model.scene.traverse(n=>{let r=n;if(!r.isMesh)return;let i=r.geometry.getAttribute(`position`)?.count??0;for(let n=0;n<i;n++)e=Math.min(e,r.getVertexPosition(n,t).applyMatrix4(r.matrixWorld).y)}),e}buildSprings(e){let t=this.model.springBoneManager;if(!this.config.mujoco_springs||!t)return null;try{return _e.build(e,t,this.model.scene)}catch{return null}}normalizeToon(e){let t=(this.config.brightness??1)/Math.max(ie(e),1);for(let{material:e,lit:n,shade:r}of this.toon)e.color.copy(n).multiplyScalar(t),e.shadeColorFactor.copy(r).multiplyScalar(t)}update(e,t,n,r){let i=this.rig;if(!i)return;let a=this.lastTime===null?0:(t-this.lastTime)/1e3,o=Math.min(a,oa);this.lastTime=t;for(let[t,n]of this.poses)n.to.copy(e[t].quaternion),n.toAt.copy(e[t].position);if(this.rootBody>=0){let t=e[this.rootBody];this.rootTo.copy(this.hipsOffset).applyQuaternion(t.quaternion).add(t.position),n&&r&&!this.attached.length&&(this.groundTo=this.groundUnder(this.rootTo,n,r)??this.groundTo)}let s=o===0||a>sa||this.rootTo.distanceTo(this.rootFrom)>ca;if(s&&this.settle(),this.springs){this.pose(i,1),this.model.update(o),this.root.updateMatrixWorld(!0),s?this.springs.reset():this.springs.step(o),this.settle();return}let c=this.model.springBoneManager,l=s?1:Math.ceil(o/aa);for(let e=1;e<=l;e++)this.pose(i,e/l),s&&(this.model.rig.update(),this.root.updateMatrixWorld(!0),c?.reset()),this.model.update(o/l),c?.update(o/l);this.settle()}settle(){for(let e of this.poses.values())e.from.copy(e.to),e.fromAt.copy(e.toAt);this.rootFrom.copy(this.rootTo),this.groundFrom=this.groundTo}groundUnder(e,t,n){let r=this.mujoco,i=ee(e).toArray();for(let e=0;e<ua;e++){let e=r.mj_ray(t,n,i,la,ne,!0,-1,ha,null);if(e<0)return null;if(i[2]-=e,t.body_rootid[t.geom_bodyid[ha[0]]]!==this.robotRoot)return i[2];i[2]-=1e-6}return null}get simulatedSpringJoints(){return this.springs?.jointCount??0}pose(e,t){for(let e of this.poses.values())e.now.slerpQuaternions(e.from,e.to,t),e.nowAt.lerpVectors(e.fromAt,e.toAt,t);if(this.attached.length)this.fix();else{Oe(e,e=>this.poses.get(e).now,this.world,this.local);for(let e=0;e<this.nodes.length;e++)this.nodes[e].quaternion.copy(this.local[e]);if(this.rootBone){this.rootNow.lerpVectors(this.rootFrom,this.rootTo,t);let n=l.lerp(this.groundFrom,this.groundTo,t);ke(e,this.rootNow,this.heightRatio,n,this.rootBone.position)}this.limbs.length&&this.reach()}}reach(){this.root.updateMatrixWorld(!0);let e=1/0;for(let t of this.limbs){let n=this.poses.get(t.target.bodyId);pa.compose($.copy(n.nowAt).add(this.root.position),n.now,fa).multiply(t.target.inBody),pa.decompose(t.at,t.turn,ma),e=Math.min(e,t.at.y)}let t=0;for(let n of this.limbs){let r=1-(n.at.y-e)/(da*n.length);r>0&&(t=Math.max(t,r*We(n.upper,n.lower,n.end,n.at)))}t>0&&this.rootBone&&(this.rootBone.getWorldPosition($),$.y-=t,this.rootBone.position.copy(this.rootBone.parent.worldToLocal($)),this.rootBone.updateMatrixWorld(!0));for(let e of this.limbs)Ue(e.upper,e.lower,e.end,e.at,e.bend,e.turn)}fix(){this.root.updateMatrixWorld(!0);for(let{bone:e,bodyId:t,inBody:n}of this.attached){let r=this.poses.get(t);pa.compose($.copy(r.nowAt).add(this.root.position),r.now,fa).multiply(n),e.matrix.copy(e.parent.matrixWorld).invert().multiply(pa),e.matrix.decompose(e.position,e.quaternion,e.scale),e.updateMatrixWorld(!0)}}dispose(){this.springs?.dispose(),this.springs=null,this.root.removeFromParent(),this.model.dispose()}};export{va as Skin};