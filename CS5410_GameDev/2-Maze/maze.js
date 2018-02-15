document.body.style.zoom = .85;
function copy(o)
{ // https://www.codementor.io/avijitgupta/deep-copying-in-js-7x6q8vh5d
  var output, v, key;
  output = Array.isArray(o) ? [] : {};
  for (key in o)
  {
    v = o[key];
    output[key] = (typeof v === 'object') ? copy(v) : v;
  }
  return output;
}

let avgFPS = {
  arr : [],
  total : 0,
  get avg() { return this.total / this.n; },
  n : 60,
  next : function(newVal) {
    this.total -= this.arr.shift();
    this.total += newVal;
    this.arr.push(newVal);
  }
}

for (i = 0; i < avgFPS.n; ++i)
{
  avgFPS.arr.push(0);
}

function rint(n)
{
  return Math.floor(Math.random() * n);
}

function makeEdge(_id, _wall, _border, _worn, _eors)
{
  return {id : _id, wall : _wall, border : _border, worn : _worn, eors : _eors};
}

function makeMaze()
{
  let maze = [];
  let edges = [];
  let id = 0;
  for (let i = 0; i < sizey; ++i)
  {
    maze.push([]);
    for (let j = 0; j < sizex; ++j)
    {
      maze[i].push({y : i, x : j, visited : false});
    }
  }

  for (let i = 0; i < sizey; ++i)
  {
    for (let j = 0; j < sizex; ++j)
    {
      /* NORTH */
      if (i === 0 || maze[i - 1][j].s === undefined)
      {
        edge = makeEdge(id++, true, i === 0 ? true : false, null, maze[i][j]);
        edges.push(edge);
        maze[i][j].n = edge;
      }
      else
      {
        maze[i][j].n = maze[i - 1][j].s;
        maze[i][j].n.eors = maze[i][j];
      }

      /* SOUTH */
      if (i === sizey - 1 || maze[i + 1][j].n === undefined)
      {
        edge = makeEdge(id++, true, i === sizey - 1 ? true : false, maze[i][j], null);
        edges.push(edge);
        maze[i][j].s = edge;
      }
      else
      {
        maze[i][j].s = maze[i + 1][j].n;
        maze[i][j].s.worn = maze[i][j];
      }

      /* WEST */
      if (j === 0 || maze[i][j - 1].e === undefined)
      {
        edge = makeEdge(id++, true, j === 0 ? true : false, maze[i][j], null);
        edges.push(edge);
        maze[i][j].w = edge;
      }
      else
      {
        maze[i][j].w = maze[i][j - 1].e;
        maze[i][j].w.eors = maze[i][j];
      }

      /* EAST */
      if (j === sizex - 1 || maze[i][j + 1].w === undefined)
      {
        edge = makeEdge(id++, true, j === sizex - 1 ? true : false, null, maze[i][j]);
        edges.push(edge);
        maze[i][j].e = edge;
        maze[i][j].w.worn = maze[i][j - 1];
      }
      else
      {
        maze[i][j].e = maze[i][j + 1].w;
        maze[i][j].e.worn = maze[i][j];
      }
    }
  }
  maze.start = maze[0][sizex - 1];
  maze.end = maze[sizey - 1][0];
  maze.character = maze.start;
  return [ maze, edges ];
}

function resetMaze()
{
  for (let i = 0; i < sizey; ++i)
    for (let j = 0; j < sizex; ++j)
      maze[i][j].visited = false;
}

function getMoves(cell)
{
  let moves = [];

  if (!cell.n.wall && !cell.n.worn.visited)
    moves.push(cell.n.worn);
  if (!cell.e.wall && !cell.e.eors.visited)
    moves.push(cell.e.eors);
  if (!cell.s.wall && !cell.s.eors.visited)
    moves.push(cell.s.eors);
  if (!cell.w.wall && !cell.w.worn.visited)
    moves.push(cell.w.worn);

  return moves;
}

function findPath(start, end)
{
  stack = [];
  let visited = 0;
  let cell = end;

  while (visited !== sizex * sizey)
  {
    if (cell === start)
    {
      stack.push(start);
      return stack;
    }

    if (!cell.visited)
    {
      ++visited
      cell.visited = true;
    }

    let moves = getMoves(cell);
    if (moves.length !== 0)
    {
      stack.push(cell);
      cell = moves[0];
    }
    else if (stack.length != 0)
    {
      let next = stack.pop();
      if (next)
        cell = next;
    }
  }
}

/***********************************************/
/*        Recursive Backtracking Algorithm     */
/***********************************************/
let visited = 0;

function breakWall(cell1, cell2)
{
  if (cell1.n === cell2.s)
    cell1.n.wall = false;
  if (cell1.e === cell2.w)
    cell1.e.wall = false;
  if (cell1.s === cell2.n)
    cell1.s.wall = false;
  if (cell1.w === cell2.e)
    cell1.w.wall = false;
}

function getUnvisited(cell)
{
  let moves = [];

  if (cell.n.worn && !cell.n.worn.visited)
    moves.push(cell.n.worn);
  if (cell.e.eors && !cell.e.eors.visited)
    moves.push(cell.e.eors);
  if (cell.s.eors && !cell.s.eors.visited)
    moves.push(cell.s.eors);
  if (cell.w.worn && !cell.w.worn.visited)
    moves.push(cell.w.worn);

  return moves;
}

function recBT(cell, stack)
{
  cell.visited = true;

  let moves = getMoves(cell);

  if (moves.length != 0)
  {
    stack.push(cell);
    let move = moves[rint(moves.length)];
    breakWall(cell, move);
    recBT(move, stack);
  }
  else if (stack.length != 0)
  {
    let next = stack.pop();
    if (next)
      recBT(next, stack);
  }
}

function recursiveBacktrack()
{
  let firstx = rint(sizex);
  let firsty = rint(sizey);

  let cell = maze[firstx][firsty];

  let stack = [];
  let visited = 0;
  while (visited != sizex * sizey)
  {
    if (!cell.visited)
    {
      ++visited;
      cell.visited = true;
    }

    let moves = getUnvisited(cell);

    if (moves.length != 0)
    {
      stack.push(cell);
      let move = moves[rint(moves.length)];
      breakWall(cell, move);
      cell = move;
    }
    else if (stack.length != 0)
    {
      let next = stack.pop();
      if (next)
        cell = next;
    }
  }
}

/***********************************************/
/*               Kruskal's Algorithm           */
/***********************************************/

function kruskals()
{
  let walls = [];
  let sets = [];

  // copy edges
  for (let i = 0; i < edges.length; ++i)
    walls[i] = edges[i];

  // put all cells in their own set
  // tell the cell what set it is in
  for (let i = 0; i < sizey; ++i)
    for (let j = 0; j < sizex; ++j)
    {
      sets.push([ maze[i][j] ]);
      maze[i][j].set = i * sizey + j;
    }

  while (walls.length !== 0)
  {
    let wallindx = rint(walls.length - 1);
    let wall = walls[wallindx];

    if (wall.worn && wall.eors) // both are defined
    {
      if (wall.worn.set !== wall.eors.set)
      {
        // get sets
        let set1 = wall.worn.set;
        let set2 = wall.eors.set;
        let keep = null;
        let conv = null;

        // choose smaller set to keep
        if (sets[set1].length > sets[set2].length)
        {
          keep = set1;
          conv = set2;
        }
        else
        {
          conv = set1;
          keep = set2;
        }

        // convert all elements of set conv to set keep
        for (let i = 0; i < sets[conv].length; ++i)
        {
          sets[conv][i].set = keep;
          sets[keep].push(sets[conv][i]);
        }

        // delete all elements of conv
        sets[conv].splice(0, sets[conv].length)

        // turn the wall off
        wall.wall = false;
      }
    }

    // remove the wall from the walls set
    walls.splice(wallindx, 1);
  }
}

/***********************************************/
/*                Prim's Algorithm             */
/***********************************************/

function addWalls(cell, list)
{
  if (!cell.n.border)
    list.push(cell.n);

  if (!cell.s.border)
    list.push(cell.s);

  if (!cell.e.border)
    list.push(cell.e);

  if (!cell.w.border)
    list.push(cell.w);
}

function prims(theMaze, edges)
{
  let inMaze = [];
  let wallList = [];

  let first = theMaze[rint(sizey)][rint(sizex)];
  first.visited = true;
  inMaze.push(first);
  addWalls(first, wallList);

  while (wallList.length !== 0)
  {
    let rnd = rint(wallList.length - 1);
    let chosenWall = wallList[rnd];
    wallList.splice(rnd, 1);
    if (chosenWall.eors.visited != chosenWall.worn.visited) // logical XOR
    {
      chosenWall.wall = false;
      if (!chosenWall.eors.visited)
      {
        chosenWall.eors.visited = true;
        addWalls(chosenWall.eors, wallList);
      }
      else // (!chosenWall.worn.visited)
      {
        chosenWall.worn.visited = true;
        addWalls(chosenWall.worn, wallList);
      }
    }
  }
  resetMaze();
}

/***********************************************/
/*                  Render Maze                */
/***********************************************/
function drawCell(cell, weight)
{
  let unitx = width / sizex;
  let unity = height / sizey;
  let off = weight / 2;

  if (cell.n.wall)
  {
    context.moveTo(cell.x * unitx - off, cell.y * unity);
    context.lineTo((cell.x + 1) * unitx + off, cell.y * unity);
  }

  if (cell.s.wall)
  {
    context.moveTo(cell.x * unitx - off, (cell.y + 1) * unity);
    context.lineTo((cell.x + 1) * unitx + off, (cell.y + 1) * unity);
  }

  if (cell.e.wall)
  {
    context.moveTo((cell.x + 1) * unitx, cell.y * unity);
    context.lineTo((cell.x + 1) * unitx, (cell.y + 1) * unity);
  }

  if (cell.w.wall)
  {
    context.moveTo(cell.x * unitx, cell.y * unity);
    context.lineTo(cell.x * unitx, (cell.y + 1) * unity);
  }
}

function drawSquareCell(cell, offx, offy, color)
{
  let unitx = (width / sizex);
  let unity = (height / sizey);

  let a = cell.x * unitx + weight;
  let b = cell.y * unity + weight;
  let c = offx * unitx - 2 * weight;
  let d = offy * unity - 2 * weight;

  context.fillStyle = color;
  context.fillRect(a, b, c, d);
}

function renderMaze()
{
  let weight = Math.floor(width / sizex / 7);
  context.beginPath();
  for (let i = 0; i < maze.length; ++i)
    for (let j = 0; j < maze[i].length; ++j)
      drawCell(maze[i][j], weight); // TODO: draw by edges

  context.moveTo(0, 0);
  context.lineTo(width, 0);
  context.lineTo(width, height);
  context.lineTo(0, height);
  context.strokeStyle = 'rgb(0, 69, 104)';
  context.closePath();

  context.lineWidth = weight;
  context.stroke();

  drawSquareCell(maze.start, 1, 1, 'red');
  drawSquareCell(maze.end, 1, 1, 'green');
}

function renderCharacter()
{
  if (maze.character)
    drawSquareCell(maze.character, .80, .80, 'orange');
}

function renderPath()
{
  for (let i = 0; i < path.length; ++i)
    drawSquareCell(path[i], .65, .65, '#091540');
}

function renderBread()
{
  let weight = Math.floor(width / sizex / 7);
  for (let i = 0; i < sizex; ++i)
    for (let j = 0; j < sizey; ++j)
      if (maze[i][j].visited)
        drawSquareCell(maze[i][j], .5, .5, '#469959');
}

function renderHint()
{
  let weight = Math.floor(width / sizex / 7);
  drawSquareCell(path[path.length - 2], .80, .80, 'yellow');
}

/***********************************************/
/*                Input Processing             */
/***********************************************/
var Key = {
  _pressed : {},

  LEFT : 37,
  UP : 38,
  RIGHT : 39,
  DOWN : 40,

  isDown : function(keyCode) { return this._pressed[keyCode]; },

  onKeydown : function(event) { this._pressed[event.keyCode] = true; },

  onKeyup : function(event) { delete this._pressed[event.keyCode]; }
};

window.addEventListener('keydown', function(event) { inputs[event.keyCode] = event.keyCode; });
// window.addEventListener('keyup', function(event) { Key.onKeyup(event); }, false);
// window.addEventListener('keydown', function(event) { Key.onKeydown(event); }, false);

function updateHighScore()
{
  if (highScores.length == 0)
  {
    highScores.push(score);
    return true;
  }

  for (let i = 0; i < highScores.length; ++i)
    if (highScores[i] < score)
    {
      highScores.splice(i, 0, score);
      if (highScores.length > 3)
        highScores.pop();
      return true;
    }

  return false;
}

/***********************************************/
/*                   Movement                  */
/***********************************************/

function moveUp()
{
  if (maze.character && !maze.character.n.wall)
  {
    maze.character = maze[maze.character.y - 1][maze.character.x];

    if (path[path.length - 2] === maze.character)
    {
      path.pop();
      if (!maze.character.visited)
        score += 100;
    }
    else
    {
      path.push(maze.character);
      if (!maze.character.visited)
        score -= 25;
    }

    maze.character.visited = true;
  }
  // showHint = false;
}

function moveRight()
{
  if (maze.character && !maze.character.e.wall)
  {
    maze.character = maze[maze.character.y][maze.character.x + 1];

    if (path[path.length - 2] === maze.character)
    {
      path.pop();
      if (!maze.character.visited)
        score += 100;
    }
    else
    {
      path.push(maze.character);
      if (!maze.character.visited)
        score -= 25;
    }

    maze.character.visited = true;
  }
  // showHint = false;
}

function moveDown()
{
  if (maze.character && !maze.character.s.wall)
  {
    maze.character = maze[maze.character.y + 1][maze.character.x];

    if (path[path.length - 2] === maze.character)
    {
      path.pop();
      if (!maze.character.visited)
        score += 100;
    }
    else
    {
      path.push(maze.character);
      if (!maze.character.visited)
        score -= 25;
    }

    maze.character.visited = true;
  }
  // showHint = false;
}

function moveLeft()
{
  if (maze.character && !maze.character.w.wall)
  {
    maze.character = maze[maze.character.y][maze.character.x - 1];

    if (path[path.length - 2] === maze.character)
    {
      path.pop();
      if (!maze.character.visited)
        score += 100;
    }
    else
    {
      path.push(maze.character);
      if (!maze.character.visited)
        score -= 25;
    }

    maze.character.visited = true;
  }
  // showHint = false;
}

function cheatMove()
{
  if (!maze.character)
    return;
  maze.character = path[path.length - 2];
  maze.character.visited = true;
  path.pop();
}

function moveCharacter(input)
{
  // if (Key.isDown(Key.UP))
  // moveUp();
  // if (Key.isDown(Key.LEFT))
  // moveLeft();
  // if (ey.isDown(Key.DOWN))
  // moveDown();
  // if (Key.isDown(Key.RIGHT))
  // moveRight();
}

/***********************************************/
/*                    Toggles                  */
/***********************************************/

function toggleScore()
{
  showScore = !showScore;
  if (showScore)
    document.getElementById('score').style.display = 'inline';
  else
    document.getElementById('score').style.display = 'none';
}
function toggleBread()
{
  showBread = !showBread;
}

function togglePath()
{
  showPath = !showPath;
}

function toggleHint()
{
  showHint = !showHint;
}

/***********************************************/
/*                   Game Loop                 */
/***********************************************/

let canvas = null;
let context = null;
let width = null;
let height = null;
let sizex = 3;
let sizey = 3;
let maze = null;
let edges = null;
let path = null;
let weight = null;
let prevTime = performance.now();
let inputs = {};
let keyBind = {
  'P' : togglePath,
  'B' : toggleBread,
  'H' : toggleHint,
  'Y' : toggleScore,
  'I' : moveUp,
  'J' : moveLeft,
  'K' : moveDown,
  'L' : moveRight,
  'W' : moveUp,
  'A' : moveLeft,
  'S' : moveDown,
  'D' : moveRight,
  'UP' : moveUp,
  'LEFT' : moveLeft,
  'DOWN' : moveDown,
  'RIGHT' : moveRight,
  'C' : cheatMove,
};
let gameWon = false;
let newGame = false;
let showBread = false;
let showPath = false;
let showHint = false;
let showScore = true;
let doTimer = false;
let time = 0.0;
let score = 0.0;
let highScores = null;

function update(dTime)
{
  avgFPS.next(1000 / dTime);
  if (doTimer)
    time += dTime;
  if (maze.character === maze.end)
  {
    gameWon = true;
    doTimer = false;
    showHint = false;
    showPath = false;
    showBread = false;
    path.pop();
    if (updateHighScore())
      window.alert('HIGH SCORE! ' + score + '\n\n' + highScores.reduce((a, v) => a + v + '\n', ''));
    else
      window.alert('YOUR SCORE: ' + score + '\n\n' + highScores.reduce((a, v) => a + v + '\n', ''));
  }
}

function processInput()
{
  for (input in inputs)
  {
    let key = keyCodeToChar[inputs[input]];
    console.log(key);
    if (key in keyBind)
    {
      keyBind[key]();
    }
  }
  inputs = {};
}

function render()
{
  context.clear();

  if (!gameWon)
  {
    renderMaze();
    renderCharacter();

    if (showHint)
      renderHint();

    if (showPath)
      renderPath();

    if (showBread)
      renderBread();
  }
  document.getElementById('fps').innerHTML = 'FPS: ' + avgFPS.avg.toFixed(2);
  document.getElementById('timer').innerHTML = (time / 1000).toFixed(2);
  document.getElementById('score').innerHTML = score;

  // renderCharacter();
}

function gameloop()
{
  let curTime = performance.now();
  let dTime = curTime - prevTime;
  prevTime = curTime;
  if (!gameWon)
  {
    processInput();
  }
  update(dTime);
  render();

  if (gameWon && !newGame)
  {
    // window.alert('YOU WIN')
    console.log('YOU WIN');
    gameWon = false;
    maze.character = null;
  }

  if (newGame)
    start();
  else
    requestAnimationFrame(gameloop);
}

function start()
{
  if (sizex > sizey)
  {
    width = 1000;
    height = sizey / sizex * width;
  }
  else
  {
    height = 1000;
    width = sizex / sizey * height;
  }

  weight = Math.floor(width / sizex / 7);

  canvas.width = width
  canvas.height = height;

  [maze, edges] = makeMaze();

  if (document.getElementById('prims').checked)
    prims(maze);
  else if (document.getElementById('kruskals').checked)
    kruskals();
  else if (document.getElementById('recursiveBT').checked)
    recursiveBacktrack();

  path = [];

  resetMaze();

  path = findPath(maze.start, maze.end);

  resetMaze();
  maze.character.visited = true;

  console.log('starting game');

  showBread = false;
  showPath = false;
  showHint = false;

  newGame = false;
  gameWon = false;
  doTimer = true;
  time = 0.0;
  score = 0.0;

  prevTime = performance.now();
  requestAnimationFrame(gameloop);
}

function init()
{
  canvas = document.getElementById('canvas-main');
  context = canvas.getContext('2d');
  document.getElementById('recursiveBT').checked = true;
  highScores = [];

  CanvasRenderingContext2D.prototype.clear = function() {
    this.save();
    this.setTransform(1, 0, 0, 1, 0, 0);
    this.clearRect(0, 0, canvas.width, canvas.height);
    this.restore();
  };

  console.log('game initialized');

  start();
}

/***********************************************/
/*               Change Maze Size              */
/***********************************************/

function standard(m, n)
{
  sizex = m;
  sizey = n;
  newGame = true;
}

function custom()
{
  let tmpx = Math.abs(document.getElementById('sizex').value);
  let tmpy = Math.abs(document.getElementById('sizey').value);
  if (tmpx == 0 || tmpy == 0)
  {
    window.alert('maze must have nonzero width and height');
    return;
  }
  // else if (tmpx > 150 || tmpy > 150)
  // if (!window.confirm('Woah! That\'s a pretty big maze, proceed at your own risk!'))
  // return;

  sizex = tmpx;
  sizey = tmpy;
  newGame = true;
}

function showHighScores()
{
  alert(highScores.reduce((a, v) => a + v + '\n', ''));
}

let keyCodeToChar = {
  8 : 'Backspace',
  9 : 'Tab',
  13 : 'Enter',
  16 : 'Shift',
  17 : 'Ctrl',
  18 : 'Alt',
  19 : 'Pause/Break',
  20 : 'Caps Lock',
  27 : 'Esc',
  32 : 'Space',
  33 : 'Page Up',
  34 : 'Page Down',
  35 : 'End',
  36 : 'Home',
  37 : 'Left',
  38 : 'Up',
  39 : 'Right',
  40 : 'Down',
  45 : 'Insert',
  46 : 'Delete',
  48 : '0',
  49 : '1',
  50 : '2',
  51 : '3',
  52 : '4',
  53 : '5',
  54 : '6',
  55 : '7',
  56 : '8',
  57 : '9',
  65 : 'A',
  66 : 'B',
  67 : 'C',
  68 : 'D',
  69 : 'E',
  70 : 'F',
  71 : 'G',
  72 : 'H',
  73 : 'I',
  74 : 'J',
  75 : 'K',
  76 : 'L',
  77 : 'M',
  78 : 'N',
  79 : 'O',
  80 : 'P',
  81 : 'Q',
  82 : 'R',
  83 : 'S',
  84 : 'T',
  85 : 'U',
  86 : 'V',
  87 : 'W',
  88 : 'X',
  89 : 'Y',
  90 : 'Z',
  91 : 'Windows',
  93 : 'Right Click',
  96 : 'Numpad 0',
  97 : 'Numpad 1',
  98 : 'Numpad 2',
  99 : 'Numpad 3',
  100 : 'Numpad 4',
  101 : 'Numpad 5',
  102 : 'Numpad 6',
  103 : 'Numpad 7',
  104 : 'Numpad 8',
  105 : 'Numpad 9',
  106 : 'Numpad *',
  107 : 'Numpad +',
  109 : 'Numpad -',
  110 : 'Numpad .',
  111 : 'Numpad /',
  112 : 'F1',
  113 : 'F2',
  114 : 'F3',
  115 : 'F4',
  116 : 'F5',
  117 : 'F6',
  118 : 'F7',
  119 : 'F8',
  120 : 'F9',
  121 : 'F10',
  122 : 'F11',
  123 : 'F12',
  144 : 'Num Lock',
  145 : 'Scroll Lock',
  182 : 'My Computer',
  183 : 'My Calculator',
  186 : ';',
  187 : '=',
  188 : ',',
  189 : '-',
  190 : '.',
  191 : '/',
  192 : '`',
  219 : '[',
  220 : '\\',
  221 : ']',
  222 : '\''
};

let keyCharToCode = {
  'Backspace' : 8,
  'Tab' : 9,
  'Enter' : 13,
  'Shift' : 16,
  'Ctrl' : 17,
  'Alt' : 18,
  'Pause/Break' : 19,
  'Caps Lock' : 20,
  'Esc' : 27,
  'Space' : 32,
  'Page Up' : 33,
  'Page Down' : 34,
  'End' : 35,
  'Home' : 36,
  'Left' : 37,
  'Up' : 38,
  'Right' : 39,
  'Down' : 40,
  'Insert' : 45,
  'Delete' : 46,
  '0' : 48,
  '1' : 49,
  '2' : 50,
  '3' : 51,
  '4' : 52,
  '5' : 53,
  '6' : 54,
  '7' : 55,
  '8' : 56,
  '9' : 57,
  'A' : 65,
  'B' : 66,
  'C' : 67,
  'D' : 68,
  'E' : 69,
  'F' : 70,
  'G' : 71,
  'H' : 72,
  'I' : 73,
  'J' : 74,
  'K' : 75,
  'L' : 76,
  'M' : 77,
  'N' : 78,
  'O' : 79,
  'P' : 80,
  'Q' : 81,
  'R' : 82,
  'S' : 83,
  'T' : 84,
  'U' : 85,
  'V' : 86,
  'W' : 87,
  'X' : 88,
  'Y' : 89,
  'Z' : 90,
  'Windows' : 91,
  'Right Click' : 93,
  'Numpad 0' : 96,
  'Numpad 1' : 97,
  'Numpad 2' : 98,
  'Numpad 3' : 99,
  'Numpad 4' : 100,
  'Numpad 5' : 101,
  'Numpad 6' : 102,
  'Numpad 7' : 103,
  'Numpad 8' : 104,
  'Numpad 9' : 105,
  'Numpad *' : 106,
  'Numpad +' : 107,
  'Numpad -' : 109,
  'Numpad .' : 110,
  'Numpad /' : 111,
  'F1' : 112,
  'F2' : 113,
  'F3' : 114,
  'F4' : 115,
  'F5' : 116,
  'F6' : 117,
  'F7' : 118,
  'F8' : 119,
  'F9' : 120,
  'F10' : 121,
  'F11' : 122,
  'F12' : 123,
  'Num Lock' : 144,
  'Scroll Lock' : 145,
  'My Computer' : 182,
  'My Calculator' : 183,
  ';' : 186,
  '=' : 187,
  ',' : 188,
  '-' : 189,
  '.' : 190,
  '/' : 191,
  '`' : 192,
  '[' : 219,
  '\\' : 220,
  ']' : 221,
  '\'' : 222
};
