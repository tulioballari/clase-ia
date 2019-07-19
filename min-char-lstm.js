const tf = require('@tensorflow/tfjs-node');

const data = readData();
// All unique characters / entities in the data set.
const data_size = data.length;
const chars = [...new Set(data)];

const V = vocab_size = chars.length;
console.log('data has %d characters, %d unique.', data_size, vocab_size);

// Each character in the vocabulary gets a unique integer index assigned, in the
// half-open interval [0:N). These indices are useful to create one-hot encoded
// vectors that represent characters in numerical computations.
const char_to_ix = chars.reduce((accum, char, index) => {
  accum[char] = index;
  return accum;
}, {});
const ix_to_char = chars;
//console.log('char_to_ix', char_to_ix)
//console.log('ix_to_char', ix_to_char)

// Hyperparameters.

// Size of hidden state vectors; applies to h and c.
const H = hidden_size = 100;
const seq_length = 16 // number of steps to unroll the LSTM for
const learning_rate = 0.1

// The input x is concatenated with state h, and the joined vector is used to
// feed into most blocks within the LSTM cell. The combined height of the column
// vector is HV.
const HV = H + V
console.log('HV %d H %d V %d', HV, H, V);

// Stop when processed this much data
const MAX_DATA = 1000000;

// Model parameters/weights -- these are shared among all steps. Weights
// initialized randomly; biases initialized to 0.
// Inputs are characters one-hot encoded in a vocab-sized vector.
// Dimensions: H = hidden_size, V = vocab_size, HV = hidden_size + vocab_size

let Wf = tf.tidy(() => tf.randomUniform([H, HV]).mul(tf.scalar(0.01)));
let bf = tf.zeros([H, 1]);

let Wi = tf.tidy(() => tf.randomUniform([H, HV]).mul(tf.scalar(0.01)));
let bi = tf.zeros([H, 1]);

let Wcc = tf.tidy(() => tf.randomUniform([H, HV]).mul(tf.scalar(0.01)));
let bcc = tf.zeros([H, 1]);

let Wo = tf.tidy(() => tf.randomUniform([H, HV]).mul(tf.scalar(0.01)));
let bo = tf.zeros([H, 1]);

let Wy = tf.tidy(() => tf.randomUniform([V, H]).mul(tf.scalar(0.01)));
let by = tf.zeros([V, 1])

// Uncomment this to run gradient checking instead of training
//basicGradCheck();
//process.exit();

// n is the iteration counter; p is the input sequence pointer, at the beginning
// of each step it points at the sequence in the input that will be used for
// training this iteration.
let n = 0;
let p = 0;

// Memory variables for Adagrad.
let mWf = tf.zerosLike(Wf)
let mbf = tf.zerosLike(bf)
let mWi = tf.zerosLike(Wi)
let mbi = tf.zerosLike(bi)
let mWcc = tf.zerosLike(Wcc)
let mbcc = tf.zerosLike(bcc)
let mWo = tf.zerosLike(Wo)
let mbo = tf.zerosLike(bo)
let mWy = tf.zerosLike(Wy)
let mby = tf.zerosLike(by)
let smooth_loss = -Math.log(1.0 / V) * seq_length;

let hprev, cprev;
while (p < MAX_DATA) {

  // Prepare inputs (we're sweeping from left to right in steps seq_length long
  if (n === 0 || p + seq_length + 1 >= data.length) {
    // Reset RNN memory
    hprev = tf.zeros([H, 1]);
    cprev = tf.zeros([H, 1])
    p = 0 // go from start of data
  }

  // In each step we unroll the RNN for seq_length cells, and present it with
  // seq_length inputs and seq_length target outputs to learn.
  const inputs = data.substring(p, p + seq_length).split('').map(ch => char_to_ix[ch]);
  const targets = data.substring(p + 1, p + seq_length + 1).split('').map(ch => char_to_ix[ch]);

  // Sample from the model now and then.
  if (n % 1000 == 0) {
    let sample_ix = sample(hprev, cprev, inputs[0], 200);
    let txt = sample_ix.map(ix => ix_to_char[ix]).join('');
    console.log('----\n %s \n----', txt);
  }

  // Forward seq_length characters through the RNN and fetch gradient.
  let loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby;
  [loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby, hprev, cprev] = lossFun(inputs, targets, hprev, cprev);
  smooth_loss = smooth_loss * 0.999 + loss * 0.001;
  if (n % 200 == 0) {
    console.log('iter %d (p=%d), loss %f', n, p, smooth_loss);
    //console.log('tf.memory:', tf.memory());
  }

  // Perform parameter update with Adagrad.
  [Wf,mWf] = paramUpdates(Wf,dWf,mWf);
  [bf,mbf] = paramUpdates(bf,dbf,mbf);
  [Wi,mWi] = paramUpdates(Wi,dWi,mWi);
  [bi,mbi] = paramUpdates(bi,dbi,mbi);
  [Wcc,mWcc] = paramUpdates(Wcc,dWcc,mWcc);
  [bcc,mbcc] = paramUpdates(bcc,dbcc,mbcc);
  [Wo,mWo] = paramUpdates(Wo,dWo,mWo);
  [bo,mbo] = paramUpdates(bo,dbo,mbo);
  [Wy,mWy] = paramUpdates(Wy,dWy,mWy);
  [by,mby] = paramUpdates(by,dby,mby);

  p += seq_length;
  n += 1;
}

console.log('tf.memory:', tf.memory());

function paramUpdates(param, dparam, mem) {
  return tf.tidy(() => {
    //mem += dparam * dparam
    //param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    let mem1 = mem.add(dparam.mul(dparam));
    let param1 = param.add(tf.scalar(-learning_rate).mul(dparam).div(tf.sqrt(mem1.add(tf.scalar(1e-8)))));
    tf.dispose([param, dparam, mem]);
    return [param1, mem1];
  });
}

// Sample a sequence of integers from the model.
// Runs the LSTM in forward mode for n steps; seed_ix is the seed letter for
// the first time step, h and c are the memory state. Returns a sequence of
// letters produced by the model (indices).
function sample(h, c, seed_ix, n) {
  return tf.tidy(() => {
    const ixes = []
    let p;
    let ix = seed_ix;
    for (let t = 0; t < n; t++) {
      // Run the forward pass only.
      ({p, h, c} = forward(ix, h, c));

      // Sample from the distribution produced by softmax.
      ix = choice(p.flatten().arraySync());
      ixes.push(ix)
    }
    return ixes
  });
}

function forward(ix, h, c) {
  return tf.tidy(() => {
    // Prepare a one-hot encoded vector of
    // shape (V, 1). inputs[t] is the index where the 1 goes.
    const x = tf.buffer([V, 1]);
    x.set(1, ix, 0);

    // hprev and xs are column vector;
    // concat them together into a "taller"
    // column vector - first the elements of x, then h.
    const xh = tf.concat([x.toTensor(), h]);

    // Gates f, i and o.
    const fg = tf.sigmoid(tf.dot(Wf, xh).add(bf));
    const ig = tf.sigmoid(tf.dot(Wi, xh).add(bi));
    const og = tf.sigmoid(tf.dot(Wo, xh).add(bo));

    // Candidate cc.
    const cc = tf.tanh(tf.dot(Wcc, xh).add(bcc));

    // This step's h and c.
    c = fg.mul(c).add(ig.mul(cc));
    h = tf.tanh(c).mul(og);

    // Softmax for output.
    const y = tf.dot(Wy, h).add(by);
    const p = tf.exp(y).div(tf.sum(tf.exp(y)));

    return {p, xh, h, c, fg, ig, og, cc, y};
  });
}

function choice(ps) {
  const random = Math.random();
  let ix = 0;
  let pAcum = 0;
  for (const p of ps) {
    pAcum = pAcum + p;

    if (random <= pAcum) {
      return ix;
    }
    ix++;
  }
}

// Runs forward and backward passes through the RNN.
// inputs, targets: Lists of integers. For some i, inputs[i] is the input
// character (encoded as an index into the ix_to_char map) and
//  targets[i] is the corresponding next character in the training data (similarly encoded).
// hprev: Hx1 array of initial hidden state
// cprev: Hx1 array of initial hidden state
// returns: loss, gradients on model parameters, and last hidden states
function lossFun(inputs, targets, hprev, cprev) {
  return tf.tidy(() => {
    // Caches that keep values computed in the forward pass at each time step, to
    // be reused in the backward pass.
    const cache = [];

    // Initial incoming states.
    let h = hprev;
    let c = cprev;

    let loss = 0;
    // Forward pass
    //for t in range(len(inputs)):
    for (let t = 0; t < inputs.length; t++) {
      //inputs[t] is the index where the 1 goes.
      let run = forward(inputs[t], h, c);

      cache.push(run); //run = {p, h, c, fg, ig, og, cc, y}
      h = run.h;
      c = run.c;
      // Cross-entropy loss.
      loss += -Math.log(run.p.arraySync()[targets[t]][0]);
    }
    //Initialize gradients of all weights/biases to 0.
    let dWf = tf.zerosLike(Wf);
    let dbf = tf.zerosLike(bf);
    let dWi = tf.zerosLike(Wi);
    let dbi = tf.zerosLike(bi);
    let dWcc = tf.zerosLike(Wcc);
    let dbcc = tf.zerosLike(bcc);
    let dWo = tf.zerosLike(Wo);
    let dbo = tf.zerosLike(bo);
    let dWy = tf.zerosLike(Wy);
    let dby = tf.zerosLike(by);

    // Incoming gradients for h and c; for backwards loop step these represent
    // dh[t] and dc[t]; we do truncated BPTT, so assume they are 0 initially.
    let dhnext = tf.zerosLike(cache[0].h)
    let dcnext = tf.zerosLike(cache[0].c)

    // The backwards pass iterates over the input sequence backwards.
    for (let t = inputs.length-1; t >= 0; t--) {
        // Backprop through the gradients of loss and softmax.
        let dy = cache[t].p.clone().bufferSync();
        dy.set(dy.get(targets[t], 0) - 1, targets[t], 0);//  dy[targets[t]] -= 1
        dy = dy.toTensor();

        // Compute gradients for the Wy and by parameters.
        dWy = dWy.add(tf.dot(dy, cache[t].h.transpose()));
        dby = dby.add(dy);

        // Backprop through the fully-connected layer (Wy, by) to h. Also add up
        // the incoming gradient for h from the next cell.
        const dh = tf.dot(Wy.transpose(), dy).add(dhnext);

        // Backprop through multiplication with output gate; here "dtanh" means
        // the gradient at the output of tanh.
        const dctanh = cache[t].og.mul(dh);

        // Backprop through the tanh function; since cs[t] branches in two
        // directions we add dcnext too.
        // dc = dctanh * (1 - np.tanh(cs[t]) ** 2) + dcnext
        const dc = dctanh.mul(
          tf.scalar(1).sub(tf.square(tf.tanh(cache[t].c)))
        ).add(dcnext);

        // Backprop through multiplication with the tanh; here "dhogs" means
        // the gradient at the output of the sigmoid of the output gate. Then
        // backprop through the sigmoid itself (ogs[t] is the sigmoid output).
        const dhogs = dh.mul(tf.tanh(cache[t].c));

        // dho = dhogs * ogs[t] * (1 - ogs[t])
        const dho = dhogs.mul(cache[t].og).mul(tf.scalar(1).sub(cache[t].og));

        // Compute gradients for the output gate parameters.
        dWo = dWo.add( tf.dot(dho, cache[t].xh.transpose()));
        dbo = dbo.add(dho);

        // Backprop dho to the xh input.
        const dxh_from_o = tf.dot(Wo.transpose(), dho);

        // Backprop through the forget gate: sigmoid and elementwise mul.
        // dhf = cs[t-1] * dc * fgs[t] * (1 - fgs[t])
        const prevC = cache[t-1]? cache[t-1].c :cprev;
        const dhf = prevC.mul(dc).mul(cache[t].fg).mul(tf.scalar(1).sub(cache[t].fg));
        dWf = dWf.add(tf.dot(dhf, cache[t].xh.transpose()));
        dbf = dbf.add(dhf);
        const dxh_from_f = tf.dot(Wf.transpose(), dhf);

        // Backprop through the input gate: sigmoid and elementwise mul.
        const dhi = cache[t].cc.mul(dc).mul(cache[t].ig).mul(tf.scalar(1).sub(cache[t].ig));
        dWi = dWi.add(tf.dot(dhi, cache[t].xh.transpose()));
        dbi = dbi.add(dhi);
        const dxh_from_i = tf.dot(Wi.transpose(), dhi)

        const dhcc = cache[t].ig.mul(dc).mul(tf.scalar(1).sub(cache[t].cc.square()));
        dWcc = dWcc.add(tf.dot(dhcc, cache[t].xh.transpose()));
        dbcc = dbcc.add(dhcc);
        dxh_from_cc = tf.dot(Wcc.transpose(), dhcc)

        // Combine all contributions to dxh, and extract the gradient for the
        // h part to propagate backwards as dhnext.
        const dxh = dxh_from_o.add(dxh_from_f).add(dxh_from_i).add(dxh_from_cc);
        dhnext = dxh.slice(V);

        // dcnext from dc and the forget gate.
        dcnext = cache[t].fg.mul(dc);
    }

    // Gradient clipping to the range [-5, 5].
    dWf = dWf.clipByValue(-5, 5);
    dbf = dbf.clipByValue(-5, 5);
    dWi = dWi.clipByValue(-5, 5);
    dbi = dbi.clipByValue(-5, 5);
    dWcc = dWcc.clipByValue(-5, 5);
    dbcc = dbcc.clipByValue(-5, 5);
    dWo = dWo.clipByValue(-5, 5);
    dbo = dbo.clipByValue(-5, 5);
    dWy = dWy.clipByValue(-5, 5);
    dby = dby.clipByValue(-5, 5);

    let prevH, prevC;
    if (cache[inputs.length-1]) {
      prevH = cache[inputs.length-1].h;
      prevC = cache[inputs.length-1].h;
      tf.dispose([hprev,cprev]);
    } else {
      prevH = hprev;
      prevC = cprev;
    }

    return [loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby, prevH, prevC]

  });
}

function readData() {
  const fs = require('fs');

  const filename = process.argv[2];
  const data_raw = fs.readFileSync(filename, 'utf8');

  let head = data_raw.indexOf('***')
  head = data_raw.indexOf('***', head + 3) + 3;
  let tail = data_raw.indexOf('***', head);

  const data = data_raw.substring(head, tail).trim();
  console.log(data.substr(0, 30));
  console.log('***');
  console.log(data.substr(-30));

  return data;
}
