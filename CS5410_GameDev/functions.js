var person = {
  name: 'joe',
  age: 24
};

function report(thing) {
  if(thing != undefined){
    console.log(thing.name);
  }
} // no semicolon

report(person);
