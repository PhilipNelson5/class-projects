Engine.ParticleSystem = (function (graphics) {
  'use-strict'
  let effects = [];

  /**
   * updates all active effects
   * @param {double} dt - the delta time or time since the last frame
   * @return {undefined}
   */
  function update(dt) {
    let keepMe = [];

    for(let i = 0; i < effects.length; ++i) {
      effects[i].update(dt);
      effects[i].duration -= dt
      if (!effects[i].done)
        keepMe.push(effects[i]);
    }

    effects.length = 0;
    effects = keepMe;
  }

  /**
   * renders all active effects
   * @return {undefined}
   */
  function render() {
    for(let i = 0; i < effects.length; ++i)
      effects[i].render();
  }

  /**
   * clears all active effects
   * @return {undefined}
   */
  function clear() {
    effects.length = 0;
  }

  /**
   * @typedef {Object} pos
   * @property {double} x The X Coordinate
   * @property {double} y The Y Coordinate
   */

  /**
   * @param {double} pos1 - is the start/tail of the vector
   * @param {double} pos2 - is the end/head of the vector
   * @return {pos} - a unit vector in the direction from pos1 to pos2
   */
  function makeVector(pos1, pos2) {
    let pos = {x: pos2.x-pos1.x, y: pos2.y-pos1.y}
    let mag = Math.sqrt(pos.x*pos.x + pos.y*pos.y)
    pos.x /= mag;
    pos.y /= mag;
    return pos;
  }

  /**
   * @param {object} spec {
   * duration     : 50 ,                      // duration to create particles
   * fade         : 500,                      // fade at this many ms left in particle life
   * fill         : {w: 50, h: 10},           // The area to fill with the effect
   * image        : '/textures/image',        // path to image
   * lifetime     : {mean: 1000, stdev: 250}, // ms for particle lifetime
   * particleRate : 3000,                     // particles created per second
   * position     : {x: <px>, y: e<px>},      // center x and y location
   * size         : {mean: 10, stdev: 5},     // size of particles
   * speed        : {mean: .1, stdev: .025},  // speed of particles
   * spread       : {mean: 0, stdev: 0},      // time when particle starts rendering
   * }
   * @return {undefined}
   */
  function createFill(spec) {
    let that = {done:false, duration:spec.duration};
    let particles = [];
    let image = new Image();
    image.onload = function () {
      that.render = function() {
        let p = null,
          alpha = 1;
        for (let i = 0; i < particles.length; ++i) {
          p = particles[i];
          if (p.alive >= p.spread) {
            if(p.lifetime-p.alive <= 500) {
              alpha = (p.lifetime - p.alive)/spec.fade;
            }
            else alpha = 1;
            graphics.drawImage(
              p.position,
              p.size,
              p.rotation,
              alpha,
              image
            );
          }
        }
      };
    };
    image.src = spec.image;

    that.update = function(dt) {
      if (particles.length === 0 && that.duration <= 0) {
        that.done = true;
      }
      let keepMe = [];

      let p = null;
      for (let i = 0; i < particles.length; ++i) {
        p = particles[i];
        p.alive += dt;
        p.position.x += (dt * p.speed * p.direction.x);
        p.position.y += (dt * p.speed * p.direction.y);
        p.rotation += p.speed / .5;

        if (p.alive <= p.lifetime) {
          keepMe.push(p);
        }
      }

      if (that.duration > 0)
      {
        for (let i = 0; i < spec.particleRate*dt*.001; ++i) {
          let x = Random.nextRange(spec.position.x-spec.fill.w/2, spec.position.x+spec.fill.w/2);
          let y = Random.nextRange(spec.position.y-spec.fill.h/2, spec.position.y+spec.fill.h/2);
          p = {
            position: {x,y},
            direction: makeVector(spec.position,{x,y}),
            speed: Random.nextGaussian( spec.speed.mean, spec.speed.stdev ),  // pixels per millisecond
            rotation: 0,
            lifetime: Random.nextGaussian(spec.lifetime.mean, spec.lifetime.stdev),  // milliseconds
            alive: 0,
            size: Random.nextGaussian(spec.size.mean, spec.size.stdev),
            spread: Random.nextGaussian(spec.spread.mean, spec.spread.stdev),
          };
          keepMe.push(p);
        }
      }
      particles = keepMe;
    };

    that.render = function() {};

    effects.push(that);
  }

  /**
   * @param {object} spec {
   * duration     : duration to create particles
   * fade         : fade at this many ms left in particle life
   * image        : path to image
   * lifetime     : {mean: 1000, stdev: 250} ms for particle lifetime
   * particleRate : the particles to create per second
   * position     : {x: <px>, y: e<px>} center x and y location
   * size         : {mean: 30, stdev: 5} size of the particles
   * speed        : {mean: .25, stdev: .025} speed of the particles
   * spread       : {mean: 250, stdev: 100} when the particle starts rendering
   * }
   * @return {undefined}
   */
  function createPoint(spec) {
    let that = {done:false, duration:spec.duration};
    let particles = [];
    let image = new Image();
    image.onload = function () {
      that.render = function() {
        let p = null,
          alpha = 1;
        for (let i = 0; i < particles.length; ++i) {
          p = particles[i];
          if (p.alive >= p.spread) {
            if(p.lifetime-p.alive <= 500) {
              alpha = (p.lifetime - p.alive)/spec.fade;
            }
            graphics.drawImage(
              p.position,
              p.size,
              p.rotation,
              alpha,
              image
            );
          }
        }
      };
    };
    image.src = spec.image;

    that.update = function(dt) {
      if (particles.length === 0 && that.duration <= 0) {
        that.done = true;
      }
      let keepMe = [];

      let p = null;
      for (let i = 0; i < particles.length; ++i) {
        p = particles[i];
        p.alive += dt;
        p.position.x += (dt * p.speed * p.direction.x);
        p.position.y += (dt * p.speed * p.direction.y);
        p.rotation += p.speed / .5;

        if (p.alive <= p.lifetime) {
          keepMe.push(p);
        }
      }

      if (that.duration > 0)
      {
        for (let i = 0; i < spec.particleRate*dt*.001; ++i) {
          p = {
            position: { x: spec.position.x, y: spec.position.y },
            direction: Random.nextCircleVector(),
            speed: Random.nextGaussian( spec.speed.mean, spec.speed.stdev ),  // pixels per millisecond
            rotation: 0,
            lifetime: Random.nextGaussian(spec.lifetime.mean, spec.lifetime.stdev),  // milliseconds
            alive: 0,
            size: Random.nextGaussian(spec.size.mean, spec.size.stdev),
            spread: Random.nextGaussian(spec.spread.mean, spec.spread.stdev),
          };
          keepMe.push(p);
        }
      }
      particles = keepMe;
    };

    that.render = function() {};

    effects.push(that);
  }

  /**
   * @param {object} spec {
   * duration     : duration to create particles
   * fade         : fade at this many ms left in particle life
   * image        : path to image
   * lifetime     : {mean: 1000, stdev: 250} ms for particle lifetime
   * particleRate : the particles to create per second
   * position     : {x: <px>, y: e<px>} center x and y location
   * size         : {mean: 30, stdev: 5} size of the particles
   * speed        : {mean: .25, stdev: .025} speed of the particles
   * spread       : {mean: 250, stdev: 100} when the particle starts rendering
   * }
   * @return {undefined}
   */
  function createGravityPoint(spec) {
    let that = {done:false, duration:spec.duration};
    let particles = [];
    let image = new Image();
    image.onload = function () {
      that.render = function() {
        let p = null,
          alpha = 1;
        for (let i = 0; i < particles.length; ++i) {
          p = particles[i];
          if (p.alive >= p.spread) {
            if(p.lifetime-p.alive <= 500) {
              alpha = (p.lifetime - p.alive)/spec.fade;
            }
            graphics.drawImage(
              p.s,
              p.size,
              p.rotation,
              alpha,
              image
            );
          }
        }
      };
    };
    image.src = spec.image;

    that.update = function(dt) {
      if (particles.length === 0 && that.duration <= 0) {
        that.done = true;
      }
      let keepMe = [];

      let p = null;
      let tn;
      let t;
      for (let i = 0; i < particles.length; ++i) {
        tn = performance.now();
        p = particles[i];
        t = (tn - p.t0)/1000;
        p.alive += dt;
        p.s.x += (p.v0 * p.direction.x * t) + (.5 * p.a.x * t * t);
        p.s.y += (p.v0 * p.direction.y * t) + (.5 * p.a.y * t * t);
        p.rotation += p.speed / .5;

        if (p.alive <= p.lifetime && p.s.y <= p.s0.y) {
          keepMe.push(p);
        }
      }

      if (that.duration > 0)
      {
        for (let i = 0; i < spec.particleRate*dt*.001; ++i) {
          p = {
            t0: performance.now(),
            direction: Random.nextCircleVectorRange(spec.angle.min, spec.angle.max),
            s: { x: spec.position.x, y: spec.position.y },
            s0: { x: spec.position.x, y: spec.position.y },
            v0: Random.nextGaussian( spec.vel.mean, spec.vel.stdev ),  // pixels per millisecond
            a: {x: 0, y: 9.81},
            rotation: 0,
            lifetime: Random.nextGaussian(spec.lifetime.mean, spec.lifetime.stdev),  // milliseconds
            alive: 0,
            size: Random.nextGaussian(spec.size.mean, spec.size.stdev),
            spread: Random.nextGaussian(spec.spread.mean, spec.spread.stdev),
            speed: Random.nextGaussian( spec.speed.mean, spec.speed.stdev ),  // pixels per millisecond
          };
          keepMe.push(p);
        }
      }
      particles = keepMe;
    };

    that.render = function() {};

    effects.push(that);
  }

  function explodeBrick(spec) {
    createFill({ // TODO: if time, really fill the brick with particles
      position: { x: spec.position.x, y: spec.position.y},
      speed: { mean: 0.1, stdev: 0.025},           // particle speed
      lifetime: { mean: 1000, stdev: 100 },        // particle lifetime
      size: { mean: spec.fill.w/10, stdev: spec.fill.w/13 },// particle size
      spread: { mean: 0, stdev: 0 },               // when particles begin to appear
      fill: {w: spec.fill.w, h: spec.fill.h },     // the width and height to cover
      duration: 100,                               // how long the effect lasts
      particleRate: 750,                           // particles created per second
      image: spec.texture,                         // particle texture
    });
  }

  function twinkle(spec) {
    createPoint({
      position: { x: spec.position.x, y: spec.position.y},
      duration : 100,
      fade : 500,
      fill : { w: 50, h: 10 },
      image : './textures/splat.png',
      lifetime : {mean: 1000, stdev: 250},
      particleRate : 3000,
      size : {mean: 10, stdev: 5},
      speed : {mean: .1, stdev: .1},
      spread : {mean: 0, stdev: 0},
    });
  }

  function gravity(spec) {
    createGravityPoint({
      position: { x: spec.position.x, y: spec.position.y},
      duration : 10,
      fade : 500,
      fill : { w: 50, h: 10 },
      image : './textures/splat2.png',
      lifetime : {mean: 2500, stdev: 250},
      particleRate : 2000,
      size : {mean: 10, stdev: 5},
      vel : {mean: 5, stdev: 1},
      speed : {mean: .1, stdev: .025},
      spread : {mean: 0, stdev: 0},
      angle : {min : -1, max: -3},
    });
  }

  return {
    render,
    update,
    createPoint,
    createFill,
    gravity,
    explodeBrick,
    twinkle,
    clear,
  }
}(Engine.graphics));
