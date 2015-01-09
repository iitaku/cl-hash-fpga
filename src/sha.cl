
void print_arr_u32x8(uint x[8]) {
  printf("[\n");
  for (int j=0; j<8; ++j) {
    printf(" 0x%08x", x[j]);
  }
  printf("\n");
  printf("]\n");
}

void print_arr_u32x64(uint x[64]) {
  printf("[\n");
  for (int j=0; j<64; ++j) {
    printf(" 0x%08x", x[j]);
    if (j % 8 == 7) {
      printf("\n");
    }
  }
  printf("]\n");
}

uint rotate_right(uint x, uint i) {
  return rotate(x, 32-i);
}

uint gather_in_be(uchar4 v) {
    return (((uint)v.x) << 24) | (((uint)v.y) << 16) | (((uint)v.z) << 8) | (((uint)v.w) << 0);
}

uchar4 scatter_in_be(uint v) {
  return (uchar4)((v & 0xFF000000) >> 24, (v & 0x00FF0000) >> 16, (v & 0x0000FF00) >> 8, (v & 0x000000FF) >> 0);
}

__kernel void sha256(const __global uchar *src, __global uchar *dst)
{
  const uint i = get_global_id(0);

  // first 32 bits of fractional parts of the square roots of the first 8 primes 2..19
  uint hs[] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

  // first 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311
  const uint keys[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
  };

  uchar buffer[64] = {0};

  for (int j=0; j<32; ++j) {
    buffer[j] = src[i*32+j];
  }

  // stop bits
  buffer[32] = 0x80;

  // size
  buffer[62] = 0x1;

  uint ws[64] = {0};

  // copy first 16 element
  for (int j=0; j<16; ++j) {
    ws[j] = gather_in_be((uchar4)(buffer[4*j+0], buffer[4*j+1], buffer[4*j+2], buffer[4*j+3]));
  }

  // extend left 48 elements
  for (int j=16; j<64; ++j) {
    const uint s0 = rotate_right(ws[j-15],  7) ^ rotate_right(ws[j-15], 18) ^ (ws[j-15] >>  3);
    const uint s1 = rotate_right(ws[j- 2], 17) ^ rotate_right(ws[j- 2], 19) ^ (ws[j- 2] >> 10);
    ws[j] = ws[j-16] + s0 + ws[j-7] + s1;
  }

  uint a = hs[0];
  uint b = hs[1];
  uint c = hs[2];
  uint d = hs[3];
  uint e = hs[4];
  uint f = hs[5];
  uint g = hs[6];
  uint h = hs[7];

  for (int j=0; j<64; ++j) {

    const uint s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    const uint ch = (e & f) ^ (~e & g);
    const uint temp1 = h + s1 + ch + keys[j] + ws[j];
    const uint s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    const uint maj = (a & b) ^ (a & c) ^ (b & c);
    const uint temp2 = s0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  hs[0] += a;
  hs[1] += b;
  hs[2] += c;
  hs[3] += d;
  hs[4] += e;
  hs[5] += f;
  hs[6] += g;
  hs[7] += h;

  for (int j=0; j<8; ++j) {
    uchar4 v = scatter_in_be(hs[j]);
    vstore4(v, 0, &dst[(i*8+j)*4]);
  }
}
