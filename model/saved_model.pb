??
?&?%
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
	DecodeCSV
records
record_defaults2OUT_TYPE
output2OUT_TYPE"%
OUT_TYPE
list(type)(0:	
2	"
field_delimstring,"
use_quote_delimbool("
na_valuestring "
select_cols	list(int)
 
B
Equal
x"T
y"T
z
"
Ttype:
2	
?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
?
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
$
StringStrip	
input

output
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b58??

global_step/Initializer/zerosConst*
dtype0	*
value	B	 R *
_output_shapes
: *
_class
loc:@global_step
k
global_step
VariableV2*
_output_shapes
: *
shape: *
dtype0	*
_class
loc:@global_step
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
f
PlaceholderPlaceholder*
shape:?????????*#
_output_shapes
:?????????*
dtype0
d
DecodeCSV/record_defaults_0Const*
dtype0*
_output_shapes
:*
valueB
B 
x
	DecodeCSV	DecodeCSVPlaceholderDecodeCSV/record_defaults_0*
OUT_TYPE
2*#
_output_shapes
:?????????
J
StringStripStringStrip	DecodeCSV*#
_output_shapes
:?????????
S
ConstConst*
dtype0*
valueBBnokey*
_output_shapes
:
?
PlaceholderWithDefaultPlaceholderWithDefaultConst*
shape:?????????*
dtype0*#
_output_shapes
:?????????
Y
ExpandDims/dimConst*
valueB :
?????????*
_output_shapes
: *
dtype0
r

ExpandDims
ExpandDimsPlaceholderWithDefaultExpandDims/dim*
T0*'
_output_shapes
:?????????
?
Dlinear/linear_model/col_0_indicator/weights/part_0/Initializer/zerosConst*
dtype0*
valueB	?*    *
_output_shapes
:	?*E
_class;
97loc:@linear/linear_model/col_0_indicator/weights/part_0
?
2linear/linear_model/col_0_indicator/weights/part_0VarHandleOp*
shape:	?*C
shared_name42linear/linear_model/col_0_indicator/weights/part_0*
dtype0*
_output_shapes
: *E
_class;
97loc:@linear/linear_model/col_0_indicator/weights/part_0
?
Slinear/linear_model/col_0_indicator/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp2linear/linear_model/col_0_indicator/weights/part_0*
_output_shapes
: 
?
9linear/linear_model/col_0_indicator/weights/part_0/AssignAssignVariableOp2linear/linear_model/col_0_indicator/weights/part_0Dlinear/linear_model/col_0_indicator/weights/part_0/Initializer/zeros*E
_class;
97loc:@linear/linear_model/col_0_indicator/weights/part_0*
dtype0
?
Flinear/linear_model/col_0_indicator/weights/part_0/Read/ReadVariableOpReadVariableOp2linear/linear_model/col_0_indicator/weights/part_0*E
_class;
97loc:@linear/linear_model/col_0_indicator/weights/part_0*
dtype0*
_output_shapes
:	?
?
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
valueB*    *
_output_shapes
:*
dtype0
?
'linear/linear_model/bias_weights/part_0VarHandleOp*
_output_shapes
: *
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
shape:*8
shared_name)'linear/linear_model/bias_weights/part_0
?
Hlinear/linear_model/bias_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/bias_weights/part_0*
_output_shapes
: 
?
.linear/linear_model/bias_weights/part_0/AssignAssignVariableOp'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0
?
;linear/linear_model/bias_weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0
?
Llinear/linear_model/linear_model/linear_model/col_0_indicator/ExpandDims/dimConst*
_output_shapes
: *
valueB :
?????????*
dtype0
?
Hlinear/linear_model/linear_model/linear_model/col_0_indicator/ExpandDims
ExpandDimsStringStripLlinear/linear_model/linear_model/linear_model/col_0_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
\linear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Vlinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/NotEqualNotEqualHlinear/linear_model/linear_model/linear_model/col_0_indicator/ExpandDims\linear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/ignore_value/x*'
_output_shapes
:?????????*
T0
?
Ulinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/indicesWhereVlinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/valuesGatherNdHlinear/linear_model/linear_model/linear_model/col_0_indicator/ExpandDimsUlinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/indices*#
_output_shapes
:?????????*
Tparams0*
Tindices0	
?
Ylinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/dense_shapeShapeHlinear/linear_model/linear_model/linear_model/col_0_indicator/ExpandDims*
_output_shapes
:*
T0*
out_type0	
?
Dlinear/linear_model/linear_model/linear_model/col_0_indicator/lookupStringToHashBucketFastTlinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/values*#
_output_shapes
:?????????*
num_buckets?
?
Ylinear/linear_model/linear_model/linear_model/col_0_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/col_0_indicator/SparseToDenseSparseToDenseUlinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/indicesYlinear/linear_model/linear_model/linear_model/col_0_indicator/to_sparse_input/dense_shapeDlinear/linear_model/linear_model/linear_model/col_0_indicator/lookupYlinear/linear_model/linear_model/linear_model/col_0_indicator/SparseToDense/default_value*'
_output_shapes
:?????????*
T0	*
Tindices0	
?
Klinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Mlinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/Const_1Const*
_output_shapes
: *
valueB
 *    *
dtype0
?
Klinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :?
?
Nlinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/on_valueConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
Olinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
Elinear/linear_model/linear_model/linear_model/col_0_indicator/one_hotOneHotKlinear/linear_model/linear_model/linear_model/col_0_indicator/SparseToDenseKlinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/depthNlinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/on_valueOlinear/linear_model/linear_model/linear_model/col_0_indicator/one_hot/off_value*
T0*,
_output_shapes
:??????????
?
Slinear/linear_model/linear_model/linear_model/col_0_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Alinear/linear_model/linear_model/linear_model/col_0_indicator/SumSumElinear/linear_model/linear_model/linear_model/col_0_indicator/one_hotSlinear/linear_model/linear_model/linear_model/col_0_indicator/Sum/reduction_indices*
T0*(
_output_shapes
:??????????
?
Clinear/linear_model/linear_model/linear_model/col_0_indicator/ShapeShapeAlinear/linear_model/linear_model/linear_model/col_0_indicator/Sum*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/col_0_indicator/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
?
Slinear/linear_model/linear_model/linear_model/col_0_indicator/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
?
Slinear/linear_model/linear_model/linear_model/col_0_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Klinear/linear_model/linear_model/linear_model/col_0_indicator/strided_sliceStridedSliceClinear/linear_model/linear_model/linear_model/col_0_indicator/ShapeQlinear/linear_model/linear_model/linear_model/col_0_indicator/strided_slice/stackSlinear/linear_model/linear_model/linear_model/col_0_indicator/strided_slice/stack_1Slinear/linear_model/linear_model/linear_model/col_0_indicator/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
?
Mlinear/linear_model/linear_model/linear_model/col_0_indicator/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value
B :?
?
Klinear/linear_model/linear_model/linear_model/col_0_indicator/Reshape/shapePackKlinear/linear_model/linear_model/linear_model/col_0_indicator/strided_sliceMlinear/linear_model/linear_model/linear_model/col_0_indicator/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
Elinear/linear_model/linear_model/linear_model/col_0_indicator/ReshapeReshapeAlinear/linear_model/linear_model/linear_model/col_0_indicator/SumKlinear/linear_model/linear_model/linear_model/col_0_indicator/Reshape/shape*(
_output_shapes
:??????????*
T0
?
:linear/linear_model/col_0_indicator/weights/ReadVariableOpReadVariableOp2linear/linear_model/col_0_indicator/weights/part_0*
_output_shapes
:	?*
dtype0
?
+linear/linear_model/col_0_indicator/weightsIdentity:linear/linear_model/col_0_indicator/weights/ReadVariableOp*
T0*
_output_shapes
:	?
?
Jlinear/linear_model/linear_model/linear_model/col_0_indicator/weighted_sumMatMulElinear/linear_model/linear_model/linear_model/col_0_indicator/Reshape+linear/linear_model/col_0_indicator/weights*'
_output_shapes
:?????????*
T0
?
Blinear/linear_model/linear_model/linear_model/weighted_sum_no_biasIdentityJlinear/linear_model/linear_model/linear_model/col_0_indicator/weighted_sum*'
_output_shapes
:?????????*
T0
?
/linear/linear_model/bias_weights/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0
?
 linear/linear_model/bias_weightsIdentity/linear/linear_model/bias_weights/ReadVariableOp*
_output_shapes
:*
T0
?
:linear/linear_model/linear_model/linear_model/weighted_sumBiasAddBlinear/linear_model/linear_model/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*'
_output_shapes
:?????????*
T0
y
linear/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0
d
linear/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
f
linear/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
f
linear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
linear/strided_sliceStridedSlicelinear/ReadVariableOplinear/strided_slice/stacklinear/strided_slice/stack_1linear/strided_slice/stack_2*
T0*
shrink_axis_mask*
_output_shapes
: *
Index0
\
linear/bias/tagsConst*
dtype0*
_output_shapes
: *
valueB Blinear/bias
e
linear/biasScalarSummarylinear/bias/tagslinear/strided_slice*
_output_shapes
: *
T0
?
3linear/zero_fraction/total_size/Size/ReadVariableOpReadVariableOp2linear/linear_model/col_0_indicator/weights/part_0*
_output_shapes
:	?*
dtype0
g
$linear/zero_fraction/total_size/SizeConst*
dtype0	*
value
B	 R?*
_output_shapes
: 
g
%linear/zero_fraction/total_zero/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
?
%linear/zero_fraction/total_zero/EqualEqual$linear/zero_fraction/total_size/Size%linear/zero_fraction/total_zero/Const*
T0	*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count/SwitchSwitch%linear/zero_fraction/total_zero/Equal%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: : 
?
3linear/zero_fraction/total_zero/zero_count/switch_tIdentity3linear/zero_fraction/total_zero/zero_count/Switch:1*
T0
*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count/switch_fIdentity1linear/zero_fraction/total_zero/zero_count/Switch*
_output_shapes
: *
T0

?
2linear/zero_fraction/total_zero/zero_count/pred_idIdentity%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: 
?
0linear/zero_fraction/total_zero/zero_count/ConstConst4^linear/zero_fraction/total_zero/zero_count/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpReadVariableOpNlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch*
_output_shapes
:	?*
dtype0
?
Nlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/SwitchSwitch2linear/linear_model/col_0_indicator/weights/part_02linear/zero_fraction/total_zero/zero_count/pred_id*E
_class;
97loc:@linear/linear_model/col_0_indicator/weights/part_0*
T0*
_output_shapes
: : 
?
=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
value
B	 R?*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/yConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
_output_shapes
: *
valueB	 R????*
dtype0	
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual	LessEqual=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeDlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/SwitchSwitchBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqualBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
_output_shapes
: : *
T0

?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_tIdentityFlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_fIdentityDlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_idIdentityBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros*
_output_shapes
:	?*
T0
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*
T0**
_output_shapes
:	?:	?*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastCastTlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes
:	?*

SrcT0
*

DstT0
?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
Ylinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_countSumPlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastQlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/CastCastYlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*

DstT0	*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes
:	?*
T0
?
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp*
T0**
_output_shapes
:	?:	?
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastCastVlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*
_output_shapes
:	?*

SrcT0

?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/MergeMerge[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countBlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Olinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/subSub=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeClinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastCastOlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1Cast=linear/zero_fraction/total_zero/zero_count/zero_fraction/Size*

SrcT0	*

DstT0*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truedivRealDivPlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastRlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Alinear/zero_fraction/total_zero/zero_count/zero_fraction/fractionIdentitySlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
2linear/zero_fraction/total_zero/zero_count/ToFloatCast9linear/zero_fraction/total_zero/zero_count/ToFloat/Switch*

DstT0*
_output_shapes
: *

SrcT0	
?
9linear/zero_fraction/total_zero/zero_count/ToFloat/SwitchSwitch$linear/zero_fraction/total_size/Size2linear/zero_fraction/total_zero/zero_count/pred_id*7
_class-
+)loc:@linear/zero_fraction/total_size/Size*
T0	*
_output_shapes
: : 
?
.linear/zero_fraction/total_zero/zero_count/mulMulAlinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction2linear/zero_fraction/total_zero/zero_count/ToFloat*
T0*
_output_shapes
: 
?
0linear/zero_fraction/total_zero/zero_count/MergeMerge.linear/zero_fraction/total_zero/zero_count/mul0linear/zero_fraction/total_zero/zero_count/Const*
N*
T0*
_output_shapes
: : 
?
)linear/zero_fraction/compute/float32_sizeCast$linear/zero_fraction/total_size/Size*

DstT0*
_output_shapes
: *

SrcT0	
?
$linear/zero_fraction/compute/truedivRealDiv0linear/zero_fraction/total_zero/zero_count/Merge)linear/zero_fraction/compute/float32_size*
_output_shapes
: *
T0
|
)linear/zero_fraction/zero_fraction_or_nanIdentity$linear/zero_fraction/compute/truediv*
_output_shapes
: *
T0
?
$linear/fraction_of_zero_weights/tagsConst*
dtype0*0
value'B% Blinear/fraction_of_zero_weights*
_output_shapes
: 
?
linear/fraction_of_zero_weightsScalarSummary$linear/fraction_of_zero_weights/tags)linear/zero_fraction/zero_fraction_or_nan*
_output_shapes
: *
T0
?
$linear/head/predictions/logits/ShapeShape:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*
_output_shapes
:
z
8linear/head/predictions/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
j
blinear/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
[
Slinear/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
?
 linear/head/predictions/logisticSigmoid:linear/linear_model/linear_model/linear_model/weighted_sum*'
_output_shapes
:?????????*
T0
?
"linear/head/predictions/zeros_like	ZerosLike:linear/linear_model/linear_model/linear_model/weighted_sum*'
_output_shapes
:?????????*
T0
x
-linear/head/predictions/two_class_logits/axisConst*
_output_shapes
: *
valueB :
?????????*
dtype0
?
(linear/head/predictions/two_class_logitsConcatV2"linear/head/predictions/zeros_like:linear/linear_model/linear_model/linear_model/weighted_sum-linear/head/predictions/two_class_logits/axis*
T0*'
_output_shapes
:?????????*
N
?
%linear/head/predictions/probabilitiesSoftmax(linear/head/predictions/two_class_logits*
T0*'
_output_shapes
:?????????
v
+linear/head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
!linear/head/predictions/class_idsArgMax(linear/head/predictions/two_class_logits+linear/head/predictions/class_ids/dimension*#
_output_shapes
:?????????*
T0
q
&linear/head/predictions/ExpandDims/dimConst*
valueB :
?????????*
_output_shapes
: *
dtype0
?
"linear/head/predictions/ExpandDims
ExpandDims!linear/head/predictions/class_ids&linear/head/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:?????????
?
linear/head/predictions/ShapeShape:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*
_output_shapes
:
u
+linear/head/predictions/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-linear/head/predictions/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-linear/head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
%linear/head/predictions/strided_sliceStridedSlicelinear/head/predictions/Shape+linear/head/predictions/strided_slice/stack-linear/head/predictions/strided_slice/stack_1-linear/head/predictions/strided_slice/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
e
#linear/head/predictions/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
e
#linear/head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
e
#linear/head/predictions/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
linear/head/predictions/rangeRange#linear/head/predictions/range/start#linear/head/predictions/range/limit#linear/head/predictions/range/delta*
_output_shapes
:
j
(linear/head/predictions/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
?
$linear/head/predictions/ExpandDims_1
ExpandDimslinear/head/predictions/range(linear/head/predictions/ExpandDims_1/dim*
_output_shapes

:*
T0
j
(linear/head/predictions/Tile/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0
?
&linear/head/predictions/Tile/multiplesPack%linear/head/predictions/strided_slice(linear/head/predictions/Tile/multiples/1*
_output_shapes
:*
N*
T0
?
linear/head/predictions/TileTile$linear/head/predictions/ExpandDims_1&linear/head/predictions/Tile/multiples*
T0*'
_output_shapes
:?????????
?
linear/head/predictions/Shape_1Shape:linear/linear_model/linear_model/linear_model/weighted_sum*
_output_shapes
:*
T0
w
-linear/head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
y
/linear/head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
y
/linear/head/predictions/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
'linear/head/predictions/strided_slice_1StridedSlicelinear/head/predictions/Shape_1-linear/head/predictions/strided_slice_1/stack/linear/head/predictions/strided_slice_1/stack_1/linear/head/predictions/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
?
*linear/head/predictions/ExpandDims_2/inputConst*
_output_shapes
:*+
value"B Bnot passwordBpassword*
dtype0
j
(linear/head/predictions/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B : 
?
$linear/head/predictions/ExpandDims_2
ExpandDims*linear/head/predictions/ExpandDims_2/input(linear/head/predictions/ExpandDims_2/dim*
_output_shapes

:*
T0
l
*linear/head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
value	B :*
dtype0
?
(linear/head/predictions/Tile_1/multiplesPack'linear/head/predictions/strided_slice_1*linear/head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:
?
linear/head/predictions/Tile_1Tile$linear/head/predictions/ExpandDims_2(linear/head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:?????????
?
1linear/head/predictions/class_string_lookup/ConstConst*+
value"B Bnot passwordBpassword*
_output_shapes
:*
dtype0
r
0linear/head/predictions/class_string_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
y
7linear/head/predictions/class_string_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
y
7linear/head/predictions/class_string_lookup/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
1linear/head/predictions/class_string_lookup/rangeRange7linear/head/predictions/class_string_lookup/range/start0linear/head/predictions/class_string_lookup/Size7linear/head/predictions/class_string_lookup/range/delta*
_output_shapes
:
?
0linear/head/predictions/class_string_lookup/CastCast1linear/head/predictions/class_string_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
w
3linear/head/predictions/class_string_lookup/Const_1Const*
valueB	 BUNK*
dtype0*
_output_shapes
: 
?
6linear/head/predictions/class_string_lookup/hash_tableHashTableV2*
	key_dtype0	*
_output_shapes
: *
value_dtype0
?
Jlinear/head/predictions/class_string_lookup/table_init/LookupTableImportV2LookupTableImportV26linear/head/predictions/class_string_lookup/hash_table0linear/head/predictions/class_string_lookup/Cast1linear/head/predictions/class_string_lookup/Const*

Tout0*	
Tin0	
?
;linear/head/predictions/hash_table_Lookup/LookupTableFindV2LookupTableFindV26linear/head/predictions/class_string_lookup/hash_table"linear/head/predictions/ExpandDims3linear/head/predictions/class_string_lookup/Const_1*

Tout0*	
Tin0	*'
_output_shapes
:?????????
f
linear/head/ShapeShape%linear/head/predictions/probabilities*
T0*
_output_shapes
:
i
linear/head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!linear/head/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
k
!linear/head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
linear/head/strided_sliceStridedSlicelinear/head/Shapelinear/head/strided_slice/stack!linear/head/strided_slice/stack_1!linear/head/strided_slice/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
{
linear/head/ExpandDims/inputConst*
dtype0*
_output_shapes
:*+
value"B Bnot passwordBpassword
\
linear/head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
linear/head/ExpandDims
ExpandDimslinear/head/ExpandDims/inputlinear/head/ExpandDims/dim*
T0*
_output_shapes

:
^
linear/head/Tile/multiples/1Const*
_output_shapes
: *
value	B :*
dtype0
?
linear/head/Tile/multiplesPacklinear/head/strided_slicelinear/head/Tile/multiples/1*
T0*
N*
_output_shapes
:
~
linear/head/TileTilelinear/head/ExpandDimslinear/head/Tile/multiples*
T0*'
_output_shapes
:?????????

initNoOp
d
init_all_tablesNoOpK^linear/head/predictions/class_string_lookup/table_init/LookupTableImportV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
|
save/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0
X
save/IdentityIdentitysave/Read/ReadVariableOp*
_output_shapes
:*
T0
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
_output_shapes
:*
T0
?
save/Read_1/ReadVariableOpReadVariableOp2linear/linear_model/col_0_indicator/weights/part_0*
_output_shapes
:	?*
dtype0
a
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes
:	?
e
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
_output_shapes
:	?*
T0
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_7cdca317c75d422893fa6657b2c9abaa/part*
_output_shapes
: *
dtype0
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:* 
valueBBglobal_step
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0
?
save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/Read_2/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
k
save/Identity_4Identitysave/Read_2/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_3/ReadVariableOpReadVariableOp2linear/linear_model/col_0_indicator/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:	?
p
save/Identity_6Identitysave/Read_3/ReadVariableOp"/device:CPU:0*
_output_shapes
:	?*
T0
e
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
_output_shapes
:	?*
T0
?
save/SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*b
valueYBWB linear/linear_model/bias_weightsB+linear/linear_model/col_0_indicator/weights*
dtype0
?
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*+
value"B B1 0,1B168 1 0,168:0,1*
dtype0
?
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_5save/Identity_7"/device:CPU:0*
dtypes
2
?
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/Identity_8Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
~
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0* 
valueBBglobal_step*
_output_shapes
:
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*
_output_shapes
:
s
save/AssignAssignglobal_stepsave/RestoreV2*
_class
loc:@global_step*
_output_shapes
: *
T0	
(
save/restore_shardNoOp^save/Assign
?
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*b
valueYBWB linear/linear_model/bias_weightsB+linear/linear_model/col_0_indicator/weights*
dtype0*
_output_shapes
:
?
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*+
value"B B1 0,1B168 1 0,168:0,1
?
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2*%
_output_shapes
::	?
a
save/Identity_9Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:

save/AssignVariableOpAssignVariableOp'linear/linear_model/bias_weights/part_0save/Identity_9"/device:CPU:0*
dtype0
i
save/Identity_10Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes
:	?
?
save/AssignVariableOp_1AssignVariableOp2linear/linear_model/col_0_indicator/weights/part_0save/Identity_10"/device:CPU:0*
dtype0
]
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"&>
save/Const:0save/Identity_8:0save/restore_all (5 @F8"A
	summaries4
2
linear/bias:0
!linear/fraction_of_zero_weights:0"?
trainable_variables??
?
4linear/linear_model/col_0_indicator/weights/part_0:09linear/linear_model/col_0_indicator/weights/part_0/AssignHlinear/linear_model/col_0_indicator/weights/part_0/Read/ReadVariableOp:0";
+linear/linear_model/col_0_indicator/weights?  "?(2Flinear/linear_model/col_0_indicator/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"?
	variables??
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
?
4linear/linear_model/col_0_indicator/weights/part_0:09linear/linear_model/col_0_indicator/weights/part_0/AssignHlinear/linear_model/col_0_indicator/weights/part_0/Read/ReadVariableOp:0";
+linear/linear_model/col_0_indicator/weights?  "?(2Flinear/linear_model/col_0_indicator/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"c
table_initializerN
L
Jlinear/head/predictions/class_string_lookup/table_init/LookupTableImportV2"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"?2
cond_context?2?2
?
4linear/zero_fraction/total_zero/zero_count/cond_text4linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_t:0 *?
2linear/zero_fraction/total_zero/zero_count/Const:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_t:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:0
?.
6linear/zero_fraction/total_zero/zero_count/cond_text_14linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_f:0*?
4linear/linear_model/col_0_indicator/weights/part_0:0
&linear/zero_fraction/total_size/Size:0
;linear/zero_fraction/total_zero/zero_count/ToFloat/Switch:0
4linear/zero_fraction/total_zero/zero_count/ToFloat:0
0linear/zero_fraction/total_zero/zero_count/mul:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_f:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual:0
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
?linear/zero_fraction/total_zero/zero_count/zero_fraction/Size:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:1
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1:0
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv:0
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:0?
4linear/linear_model/col_0_indicator/weights/part_0:0Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0e
&linear/zero_fraction/total_size/Size:0;linear/zero_fraction/total_zero/zero_count/ToFloat/Switch:02?

?

Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_textGlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0 *?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:12?

?

Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_text_1Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0*?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0"%
saved_model_main_op


group_deps*?
predict?
2
key+
PlaceholderWithDefault:0?????????
+
csv_row 
Placeholder:0?????????F
all_classes7
 linear/head/predictions/Tile_1:0?????????*
key#
ExpandDims:0?????????_
classesT
=linear/head/predictions/hash_table_Lookup/LookupTableFindV2:0?????????]
logitsS
<linear/linear_model/linear_model/linear_model/weighted_sum:0?????????F
all_class_ids5
linear/head/predictions/Tile:0?????????O
probabilities>
'linear/head/predictions/probabilities:0?????????H
	class_ids;
$linear/head/predictions/ExpandDims:0	?????????E
logistic9
"linear/head/predictions/logistic:0?????????tensorflow/serving/predict