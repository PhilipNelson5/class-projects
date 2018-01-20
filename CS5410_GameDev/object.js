// This is a JS object

// decleration of the type creates an instance
var person = {
  //properties
  first: 'john',
  'last name': 'doe',
  age: 26
};

// console.log('First Name:\t' + person.first);
// console.log('Last Name:\t' + person['last name']);
// console.log('age:\t\t' + person.age);

var person2 = {
  name: {
    first: 'jane',
    last: 'doe'
  },
  age: 25
}

// all properties have 3 intrinsic properties
//    - writable      (
//    - enumerable    ( shows up in reflection
//    - configurable  ( deletable

var circle = {
  radius: 4,
  get diameter() { return this.radius * 2},
  set diameter(val) { this.radius = val / 2}
}

console.log(circle.radius);
console.log(circle.diameter);
circle.diameter = 4;
console.log(circle.radius);
console.log(circle.diameter);

Object.defineProperty(circle, 'area', {
  value: 3.14*circle.radius*circle.radius,
  writable: false,
  enumerable: true,
  configurable: true
});

circle.circumference = 2*3.14*circle.radius;

for(var property in circle) { // reflection
  console.log(property);
  console.log(property + ": " + circle[property]);
}
