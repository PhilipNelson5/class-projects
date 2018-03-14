// ------------------------------------------------------------------
//
// This module manages persistent data for high scores
//
// ------------------------------------------------------------------
MyGame.persistence = (function () {
  'use strict';
  let highScores = {},
    previousScores = localStorage.getItem('MyGame.highScores');

  if (previousScores !== null) {
    highScores = JSON.parse(previousScores);
  }

  function add(key, value) {
    highScores[key] = value;
    localStorage['MyGame.highScores'] = JSON.stringify(highScores);
  }

  function remove(key) {
    delete highScores[key];
    localStorage['MyGame.highScores'] = JSON.stringify(highScores);
  }

  function clear() {
    for(let key in highScores) {
      delete highScores[key];
    }
    localStorage['MyGame.highScores'] = JSON.stringify(highScores);
  }

  function insertScore(score, key) {
    let oldval;
    for(let i = key; i < Object.keys(highScores).length; ++i) {
      oldval = highScores[i];
      highScores[i] = score;
      score = oldval;
    }
    if(Object.keys(highScores).length < 5) {
      add(Object.keys(highScores).length, score)
    }
    localStorage['MyGame.highScores'] = JSON.stringify(highScores);
  }

  function newScore(score) {
    for(let key in highScores) {
      if(score > highScores[key]) {
        insertScore(score, key);
        return;
      }
    }
    if(Object.keys(highScores).length < 5) {
      add(Object.keys(highScores).length, score)
    }
  }
  function report() {
    let htmlNode = document.getElementById('id-high-score-list');
    let key;

    htmlNode.innerHTML = '';
    for (key in highScores) {
      htmlNode.innerHTML += ('<li>' + highScores[key] + '</li>');
    }
  }

  return {
    add    : add,
    remove : remove,
    clear  : clear,
    report : report,
    newScore : newScore,
  };
}());
