//! GGUF file-format reader, ported to safe Rust from `ggml/src/gguf.cpp`.
//!
//! GGUF is the binary container llama.cpp uses for models. The on-disk layout
//! (little-endian throughout) is:
//!
//! ```text
//! magic "GGUF" | version:u32 | n_tensors:i64 | n_kv:i64
//! n_kv  × key-value metadata pairs
//! n_tensors × tensor-info records (name, dims, type, offset)
//! [padding to alignment] tensor data blob
//! ```
//!
//! This crate parses the header, all metadata, and all tensor-info records into
//! owned Rust types. It does not (yet) memory-map or dequantize the data blob;
//! [`Gguf::data_offset`] tells you where that blob begins. Part of the
//! in-progress Rust port of llama.cpp.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::Path;

use ggml::GgmlType;

/// `"GGUF"` little-endian magic.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Highest GGUF version this reader supports (also the version it would write).
pub const GGUF_VERSION: u32 = 3;
/// Oldest GGUF version this reader supports. v2 and v3 share the same
/// little-endian layout (64-bit counts and string/array lengths); v1's older
/// 32-bit layout is not handled yet.
pub const GGUF_VERSION_MIN: u32 = 2;
/// Default tensor-data alignment when `general.alignment` is absent.
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;
/// Metadata key overriding the tensor-data alignment.
pub const GGUF_KEY_ALIGNMENT: &str = "general.alignment";

/// Errors produced while parsing a GGUF file.
#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    /// Tried to read past the end of the buffer.
    UnexpectedEof {
        needed: usize,
        available: usize,
    },
    InvalidUtf8,
    /// A metadata value used an unknown `gguf_type` discriminant.
    UnknownValueType(u32),
    /// A metadata value was an array of arrays, which GGUF forbids.
    NestedArray,
    /// A tensor declared an unknown/removed `ggml_type`.
    UnknownTensorType(i32),
    /// A tensor declared more than `GGML_MAX_DIMS` dimensions.
    TooManyDims(u32),
    /// A length/count field was negative.
    NegativeLength(i64),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::BadMagic(m) => write!(f, "bad GGUF magic: {m:?} (expected \"GGUF\")"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            GgufError::UnexpectedEof { needed, available } => {
                write!(
                    f,
                    "unexpected end of file: needed {needed} bytes, {available} available"
                )
            }
            GgufError::InvalidUtf8 => write!(f, "string was not valid UTF-8"),
            GgufError::UnknownValueType(t) => write!(f, "unknown gguf value type: {t}"),
            GgufError::NestedArray => write!(f, "nested arrays are not permitted in GGUF"),
            GgufError::UnknownTensorType(t) => write!(f, "unknown/removed ggml tensor type: {t}"),
            GgufError::TooManyDims(n) => write!(f, "tensor has too many dimensions: {n}"),
            GgufError::NegativeLength(n) => write!(f, "negative length/count field: {n}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

type Result<T> = std::result::Result<T, GgufError>;

/// A scalar or array metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Array),
}

/// A homogeneous metadata array (GGUF arrays cannot nest).
#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl Array {
    /// Number of elements, regardless of element type.
    pub fn len(&self) -> usize {
        match self {
            Array::U8(v) => v.len(),
            Array::I8(v) => v.len(),
            Array::U16(v) => v.len(),
            Array::I16(v) => v.len(),
            Array::U32(v) => v.len(),
            Array::I32(v) => v.len(),
            Array::F32(v) => v.len(),
            Array::Bool(v) => v.len(),
            Array::String(v) => v.len(),
            Array::U64(v) => v.len(),
            Array::I64(v) => v.len(),
            Array::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Value {
    /// Borrow as a string, if this value is a `String`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as an array, if this value is an `Array`.
    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Interpret any integer scalar as `u64` (handy for hyperparameters that
    /// are written with varying widths across models).
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Value::U8(v) => *v as u64,
            Value::U16(v) => *v as u64,
            Value::U32(v) => *v as u64,
            Value::U64(v) => *v,
            Value::I8(v) if *v >= 0 => *v as u64,
            Value::I16(v) if *v >= 0 => *v as u64,
            Value::I32(v) if *v >= 0 => *v as u64,
            Value::I64(v) if *v >= 0 => *v as u64,
            Value::Bool(v) => *v as u64,
            _ => return None,
        })
    }

    /// Interpret any float scalar as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            _ => None,
        }
    }
}

/// Metadata for a single tensor (its data lives in the blob at
/// [`Gguf::data_offset`] `+` [`offset`](TensorInfo::offset)).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// Extents per dimension (`ne`), innermost first, length 1..=4.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Byte offset of this tensor within the (aligned) data blob.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements (product of `dims`).
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Size of the tensor's data in bytes, assuming the contiguous packed
    /// layout GGUF stores (`ggml_nbytes` for a contiguous tensor).
    pub fn n_bytes(&self) -> usize {
        if self.dims.is_empty() {
            return 0;
        }
        let row = self.ty.row_size(self.dims[0] as i64);
        let n_rows: u64 = self.dims[1..].iter().product();
        row * n_rows as usize
    }
}

/// A parsed GGUF file's header, metadata and tensor table.
#[derive(Debug, Clone)]
pub struct Gguf {
    version: u32,
    alignment: usize,
    data_offset: u64,
    kv: Vec<(String, Value)>,
    kv_index: HashMap<String, usize>,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
}

impl Gguf {
    /// Read and parse a GGUF file from disk.
    ///
    /// The whole file is read into memory; only the metadata is retained. (A
    /// memory-mapped, data-aware loader will come with the tensor backends.)
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        let bytes = std::fs::read(path)?;
        Gguf::from_bytes(&bytes)
    }

    /// Parse a GGUF file from an in-memory buffer.
    pub fn from_bytes(data: &[u8]) -> Result<Gguf> {
        let mut r = Reader::new(data);

        let magic = r.array4()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = r.u32()?;
        if !(GGUF_VERSION_MIN..=GGUF_VERSION).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let n_tensors = non_negative(r.i64()?)? as usize;
        let n_kv = non_negative(r.i64()?)? as usize;

        let mut kv = Vec::with_capacity(n_kv);
        let mut kv_index = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty)?;
            kv_index.insert(key.clone(), kv.len());
            kv.push((key, value));
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_index = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if n_dims as usize > ggml::GGML_MAX_DIMS {
                return Err(GgufError::TooManyDims(n_dims));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(non_negative(r.i64()?)? as u64);
            }
            let ty_raw = r.i32()?;
            let ty = GgmlType::from_i32(ty_raw).ok_or(GgufError::UnknownTensorType(ty_raw))?;
            let offset = r.u64()?;
            tensor_index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
            });
        }

        // Alignment may be overridden by metadata.
        let alignment = kv_index
            .get(GGUF_KEY_ALIGNMENT)
            .and_then(|&i| kv[i].1.as_u64())
            .filter(|&a| a != 0)
            .map(|a| a as usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // The data blob starts after the metadata, padded to `alignment` — but
        // only when the file actually contains tensors (matches gguf.cpp).
        let meta_end = r.pos();
        let data_offset = if tensors.is_empty() {
            meta_end as u64
        } else {
            align_up(meta_end, alignment) as u64
        };

        Ok(Gguf {
            version,
            alignment,
            data_offset,
            kv,
            kv_index,
            tensors,
            tensor_index,
        })
    }

    /// GGUF format version (always [`GGUF_VERSION`] for now).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Tensor-data alignment in bytes.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Byte offset at which the tensor-data blob begins.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// All metadata key-value pairs, in file order.
    pub fn metadata(&self) -> &[(String, Value)] {
        &self.kv
    }

    /// Look up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv_index.get(key).map(|&i| &self.kv[i].1)
    }

    /// Convenience: look up a string-valued metadata entry.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// Convenience: look up an integer-valued metadata entry as `u64`.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    /// All tensor-info records, in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Look up a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }
}

fn non_negative(v: i64) -> Result<i64> {
    if v < 0 {
        Err(GgufError::NegativeLength(v))
    } else {
        Ok(v)
    }
}

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align != 0);
    value.div_ceil(align) * align
}

/// A little-endian cursor over a byte slice with bounds-checked reads.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(GgufError::UnexpectedEof {
                needed: n,
                available: self.remaining(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn array4(&mut self) -> Result<[u8; 4]> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.take(4)?);
        Ok(a)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    fn u16(&mut self) -> Result<u16> {
        let mut a = [0u8; 2];
        a.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(a))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(a))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(a))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = non_negative(self.u64()? as i64)? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
    }

    /// Read a value of the given `gguf_type` discriminant.
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.i8()?),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.i16()?),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.i32()?),
            6 => Value::F32(self.f32()?),
            7 => Value::Bool(self.bool()?),
            8 => Value::String(self.string()?),
            9 => Value::Array(self.array()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.i64()?),
            12 => Value::F64(self.f64()?),
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }

    fn array(&mut self) -> Result<Array> {
        let elem_ty = self.u32()?;
        let n = non_negative(self.u64()? as i64)? as usize;
        // Cap pre-allocation by remaining bytes so a corrupt count can't OOM us
        // (every element is at least one byte on disk).
        let cap = n.min(self.remaining());

        macro_rules! read_vec {
            ($read:ident) => {{
                let mut v = Vec::with_capacity(cap);
                for _ in 0..n {
                    v.push(self.$read()?);
                }
                v
            }};
        }

        Ok(match elem_ty {
            0 => Array::U8(read_vec!(u8)),
            1 => Array::I8(read_vec!(i8)),
            2 => Array::U16(read_vec!(u16)),
            3 => Array::I16(read_vec!(i16)),
            4 => Array::U32(read_vec!(u32)),
            5 => Array::I32(read_vec!(i32)),
            6 => Array::F32(read_vec!(f32)),
            7 => Array::Bool(read_vec!(bool)),
            8 => Array::String(read_vec!(string)),
            9 => return Err(GgufError::NestedArray),
            10 => Array::U64(read_vec!(u64)),
            11 => Array::I64(read_vec!(i64)),
            12 => Array::F64(read_vec!(f64)),
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory GGUF buffer for round-trip parsing tests.
    struct Builder {
        kv: Vec<u8>,
        n_kv: i64,
        tensors: Vec<u8>,
        n_tensors: i64,
    }

    impl Builder {
        fn new() -> Self {
            Builder {
                kv: Vec::new(),
                n_kv: 0,
                tensors: Vec::new(),
                n_tensors: 0,
            }
        }

        fn push_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        fn kv_u32(mut self, key: &str, val: u32) -> Self {
            Self::push_str(&mut self.kv, key);
            self.kv.extend_from_slice(&4u32.to_le_bytes()); // gguf type UINT32
            self.kv.extend_from_slice(&val.to_le_bytes());
            self.n_kv += 1;
            self
        }

        fn kv_string(mut self, key: &str, val: &str) -> Self {
            Self::push_str(&mut self.kv, key);
            self.kv.extend_from_slice(&8u32.to_le_bytes()); // gguf type STRING
            Self::push_str(&mut self.kv, val);
            self.n_kv += 1;
            self
        }

        fn kv_str_array(mut self, key: &str, vals: &[&str]) -> Self {
            Self::push_str(&mut self.kv, key);
            self.kv.extend_from_slice(&9u32.to_le_bytes()); // gguf type ARRAY
            self.kv.extend_from_slice(&8u32.to_le_bytes()); // element type STRING
            self.kv
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                Self::push_str(&mut self.kv, v);
            }
            self.n_kv += 1;
            self
        }

        fn tensor(mut self, name: &str, dims: &[i64], ty: GgmlType, offset: u64) -> Self {
            Self::push_str(&mut self.tensors, name);
            self.tensors
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                self.tensors.extend_from_slice(&d.to_le_bytes());
            }
            self.tensors.extend_from_slice(&(ty as i32).to_le_bytes());
            self.tensors.extend_from_slice(&offset.to_le_bytes());
            self.n_tensors += 1;
            self
        }

        fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&GGUF_MAGIC);
            out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
            out.extend_from_slice(&self.n_tensors.to_le_bytes());
            out.extend_from_slice(&self.n_kv.to_le_bytes());
            out.extend_from_slice(&self.kv);
            out.extend_from_slice(&self.tensors);
            out
        }
    }

    #[test]
    fn parses_scalar_string_and_array_metadata() {
        let buf = Builder::new()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.block_count", 32)
            .kv_str_array("tokenizer.ggml.tokens", &["<s>", "</s>", "hello"])
            .build();

        let g = Gguf::from_bytes(&buf).unwrap();
        assert_eq!(g.version(), 3);
        assert_eq!(g.metadata().len(), 3);
        assert_eq!(g.get_str("general.architecture"), Some("llama"));
        assert_eq!(g.get_u64("llama.block_count"), Some(32));
        match g.get("tokenizer.ggml.tokens").and_then(Value::as_array) {
            Some(Array::String(v)) => assert_eq!(v, &["<s>", "</s>", "hello"]),
            other => panic!("expected string array, got {other:?}"),
        }
        // No tensors -> data offset is the unpadded end of metadata == file len.
        assert_eq!(g.data_offset() as usize, buf.len());
    }

    #[test]
    fn parses_tensor_info_and_aligns_data_offset() {
        let buf = Builder::new()
            .kv_u32("answer", 42)
            .tensor("token_embd.weight", &[4096, 32000], GgmlType::Q4_K, 0)
            .tensor("output_norm.weight", &[4096], GgmlType::F32, 1024)
            .build();

        let g = Gguf::from_bytes(&buf).unwrap();
        assert_eq!(g.tensors().len(), 2);

        let emb = g.tensor("token_embd.weight").unwrap();
        assert_eq!(emb.ty, GgmlType::Q4_K);
        assert_eq!(emb.dims, vec![4096, 32000]);
        assert_eq!(emb.n_elements(), 4096 * 32000);
        // 4096*32000 / 256 blocks * 144 bytes.
        assert_eq!(emb.n_bytes(), (4096 * 32000 / 256) * 144);

        let norm = g.tensor("output_norm.weight").unwrap();
        assert_eq!(norm.n_bytes(), 4096 * 4);

        // With tensors present, the data blob is padded to alignment (default 32).
        assert_eq!(g.alignment(), 32);
        assert_eq!(g.data_offset() % 32, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let err = Gguf::from_bytes(b"NOPExxxxxxxxxxxxxxxxxxxx").unwrap_err();
        assert!(matches!(err, GgufError::BadMagic(_)));
    }

    #[test]
    fn rejects_truncated_file() {
        let mut buf = Builder::new().kv_u32("answer", 42).build();
        buf.truncate(buf.len() - 2); // chop the value
        let err = Gguf::from_bytes(&buf).unwrap_err();
        assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    }

    #[test]
    fn honors_alignment_override() {
        let buf = Builder::new()
            .kv_u32(GGUF_KEY_ALIGNMENT, 64)
            .tensor("t", &[8], GgmlType::F32, 0)
            .build();
        let g = Gguf::from_bytes(&buf).unwrap();
        assert_eq!(g.alignment(), 64);
        assert_eq!(g.data_offset() % 64, 0);
    }
}
