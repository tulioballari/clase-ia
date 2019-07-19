const tf = require('@tensorflow/tfjs-node');
const _  = require('lodash');

function readData() {
  const fs = require('fs');

  const filename = process.argv[2];
  const dataRaw = fs.readFileSync(filename, 'utf8');

  let head = dataRaw.indexOf('***')
  head = dataRaw.indexOf('***', head + 3) + 3;
  let tail = dataRaw.indexOf('***', head);

  const data = dataRaw.substring(head, tail).trim();
  console.log(data.substr(0, 30));
  console.log('***');
  console.log(data.substr(-30));

  const characters = data.split('');

  let currenWord = null;
  const words = [];
  for (let ch of characters) {
    if (isLetter(ch)) {
      if (!currenWord) {
        currenWord = [];
      } 
      currenWord.push(ch.toLowerCase());
    } else if (isSpace(ch)) {
      if (currenWord) {
        words.push(currenWord.join(''));
        currenWord = null;
      }
    } else { //isSimbol
      if (currenWord) {
        words.push(currenWord.join(''));
        currenWord = null;
      }
      if (!isNumber(ch)) {
        words.push(ch);
      }
    }
  }

  return words;
}

function isLetter(ch) {
  return ch.toLowerCase() != ch.toUpperCase();
}

function isSpace(ch) {
  return /\s/.test(ch);
}

function isNumber(ch) {
  return /\d/.test(ch);
}


function createWordMap(wordArray){
  const countedWordObject = wordArray.reduce((acc, cur, i) => {
      if (acc[cur] === undefined) {
          acc[cur] = 1
      } else {
          acc[cur] += 1
      }
      return acc
  }, {})

  const arraOfshit = []
  for (let key in countedWordObject) {
      arraOfshit.push({ word: key, occurence: countedWordObject[key] })
  }

  const wordMap = _.sortBy(arraOfshit, 'occurence').reverse().map((e, i) => {
      e['code'] = i
      return e
  })

  return wordMap
}

// return a word
function fromSymbol(wordMap, symbol){
  const object = wordMap.filter(e => e.code === symbol)[0]
  return object.word
}

// return a symbol
function toSymbol(wordMap, word){
  const object = wordMap.filter(e => e.word === word)[0]
  return object.code
}

// return onehot vector, for compare with probability distribution vector
function encode(symbol){
  // console.log(symbol)
  return tf.tidy(() => {
      const symbolTensor1d = tf.tensor1d(symbol, 'int32')
      return tf.oneHot(symbolTensor1d, wordMapLength)
  })
}

// return a symbol
function decode(probDistVector){
  // It could be swithced to tf.argMax(), but I experiment with values below treshold.
  const probs = probDistVector.softmax().dataSync()
  const maxOfProbs = _.max(probs)
  const probIndexes = []

  for (let prob of probs) {
      if (prob > (maxOfProbs - 0.3)) {
          probIndexes.push(probs.indexOf(prob))
      }
  }

  return probIndexes[_.random(0, probIndexes.length - 1)]
}

// sample shape: [batch, sequence, feature], here is [1, number of words, 1]
function predict(model, samples){
  return model.predict(samples)
}

function loss(labels, predictions){
  return tf.losses.softmaxCrossEntropy(labels, predictions).mean();
}




const inputText = readData();
const preparedDataforTestSet = inputText;

// preparing data
const wordMap = createWordMap(inputText)
const wordMapLength = Object.keys(wordMap).length

console.log(`
  Number of unique words:  ${wordMapLength}
  Length of examined text: ${preparedDataforTestSet.length}
`)

const numIterations = 2000//0;
const learning_rate = 0.001;
const rnn_hidden = 64;
const examinedNumberOfWord = 6;
const endOfSeq = preparedDataforTestSet.length - (examinedNumberOfWord + 1);
const optimizer = tf.train.rmsprop(learning_rate);

let stop_training = false

// building the model
const wordVector = tf.input({ shape: [examinedNumberOfWord, 1] });
const cells = [
    tf.layers.lstmCell({ units: rnn_hidden }),
    tf.layers.lstmCell({ units: rnn_hidden }),
];
const rnn = tf.layers.rnn({ cell: cells, returnSequences: false });

const rnn_out = rnn.apply(wordVector);
const output = tf.layers.dense({ units: wordMapLength, useBias: true }).apply(rnn_out)

const model = tf.model({ inputs: wordVector, outputs: output });


// performance could be improved if toSymbol the whole set
// then random select from encodings not from array of string
const getSamples = () => {
  const startOfSeq = _.random(0, endOfSeq, false)
  const retVal = preparedDataforTestSet.slice(startOfSeq, startOfSeq + (examinedNumberOfWord + 1))
  return retVal
}



const train = async (numIterations) => {

  let lossCounter = null

  for (let iter = 0; iter < numIterations; iter++) {

      let labelProbVector
      let lossValue
      let pred
      let losse
      let samplesTensor

      const samples = getSamples().map(s => {
          return toSymbol(wordMap, s)
      })

      labelProbVector = encode(samples.splice(-1))

      if (stop_training) {
          stop_training = false
          break
      }

      // optimizer.minimize is where the training happens. 

      // The function it takes must return a numerical estimate (i.e. loss) 
      // of how well we are doing using the current state of
      // the variables we created at the start.

      // This optimizer does the 'backward' step of our training process
      // updating variables defined previously in order to minimize the
      // loss.
      lossValue = optimizer.minimize(() => {
          // Feed the examples into the model
          samplesTensor = tf.tensor(samples, [1, examinedNumberOfWord, 1])
          pred = predict(model, samplesTensor);
          losse = loss(labelProbVector, pred);
          return losse
      }, true);

      if (lossCounter === null) {
          lossCounter = lossValue.dataSync()[0]
      }
      lossCounter += lossValue.dataSync()[0]


      if (iter % 100 === 0 && iter > 50) {
          const lvdsy = lossCounter / 100
          lossCounter = 0
          console.log(`
          --------
          Step number: ${iter}
          The average loss is (last 100 steps):  ${lvdsy}
          Number of tensors in memory: ${tf.memory().numTensors}
          --------`)
      }

      // Use tf.nextFrame to not block the browser.
      //await tf.nextFrame();
      pred.dispose()
      labelProbVector.dispose()
      lossValue.dispose()
      losse.dispose()
      samplesTensor.dispose()
  }
}


const learnToGuessWord = async () => {
  console.log('TRAINING START')

  await train(numIterations);

  console.log('TRAINING IS OVER')

  const symbolCollector = _.shuffle(getSamples()).map(s => {
      return toSymbol(wordMap, s)
  })

  for (let i = 0; i < 30; i++) {
      const predProbVector = predict(model, tf.tensor(symbolCollector.slice(-examinedNumberOfWord), [1, examinedNumberOfWord, 1]))
      symbolCollector.push(decode(predProbVector));
  }

  const generatedText = symbolCollector.map(s => {
      return fromSymbol(wordMap, s)
  }).join(' ')

  console.log(generatedText)
}


learnToGuessWord();
