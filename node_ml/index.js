console.log("hello world")

// index.js

require('./app/index') //test

var svm = require('node-svm');
var so = require('stringify-object');
var Q = require('q');


var trainingFile = 'train.json';
var testingFile = 'test.json';

var clf = new svm.OneClassSVM({
    nu: 0.1,
    kernelType: svm.kernelTypes.RBF,
    gamma: 0.1,
    normalize: false,
    reduce: false,
    kFold: 1 // disable k-fold cross-validation
});

Q.all([
    svm.read(trainingFile),
    svm.read(testingFile)
]).spread(function (trainingSet, testingSet) {
    return clf.train(trainingSet)
        .progress(function(progress){
            console.log('training progress: %d%', Math.round(progress*100));
        })
        .then(function () {
            return clf.evaluate(testingSet);
        });
}).done(function (evaluationReport) {
    console.log('Accuracy against the test set:\n', so(evaluationReport));
});

// var xor = [
//     [[0, 0], 0],
//     [[0, 1], 1],
//     [[1, 0], 1],
//     [[1, 1], 0]
// ];

// // initialize a new predictor
// var clf = new svm.CSVC();

// clf.train(xor).done(function () {
//     // predict things
//     xor.forEach(function(ex){
//         var prediction = clf.predictSync(ex[0]);
//         console.log('%d XOR %d => %d', ex[0][0], ex[0][1], prediction);
//     });
// });

