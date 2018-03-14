MyGame.ParticleSystem = (function (graphics) {
  let effects = [];

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

  function render() {
    for(let i = 0; i < effects.length; ++i)
      effects[i].render();
  }

  function clear() {
    effects.length = 0;
  }

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
              alpha = (p.lifetime - p.alive)/500;
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
          p = {
            position: {
              x: Random.nextRange(spec.position.x-spec.fill.w/2, spec.position.x+spec.fill.w/2),
              y: Random.nextRange(spec.position.y-spec.fill.w/2, spec.position.y+spec.fill.w/2),
            },
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

  function createPoint(spec) {
    let that = {done:false, duration:spec.duration};
    let particles = [];
    let image = new Image();
    image.onload = function () {
      that.render = function() {
        let p = null;
        for (let i = 0; i < particles.length; ++i) {
          p = particles[i];
          if (p.alive >= p.spread) {
            graphics.drawImage(
              p.position,
              p.size,
              p.rotation,
              image);
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

return {
  render : render,
  update : update,
  createPoint : createPoint,
  createFill : createFill,
  explodeBrick : explodeBrick,
  clear : clear,
}
}(MyGame.graphics));
