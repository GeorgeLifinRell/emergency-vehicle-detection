var vehicleId = 0
let mapOffsetX = document.querySelector('.mapElem').getBoundingClientRect().x
let mapOffsetY = document.querySelector('.mapElem').getBoundingClientRect().y

const ts0 = document.getElementById('ts0')
const ts1 = document.getElementById('ts1')
const ts2 = document.getElementById('ts2')
const ts3 = document.getElementById('ts3')

var signal
var redLight
var yellowLight
var greenLight
var trafficPassed = false
var signalState = [0, 0, 0, 0]

var prevVehicle = new Array(8)

function getSignalState(signal) {
  return signalState[signal]
}

async function createVehicle(mode, option1) {
  let source
  let trackId
  let trackElem
  let transform
  let vehicleType
  let directions = ['up', 'right', 'down', 'left']
  let direction
  if (mode == 0) {
    let sources = ['bus.png', 'truck.png', 'bike.png', 'car.png']
    vehicleType = Math.floor(Math.random() * sources.length)
    source = sources[vehicleType]
    direction = directions[Math.floor(Math.random() * directions.length)]
  } else if(mode== 1){
    vehicleType = 4
    source = "ambulance.png"
    direction = option1
  }

  let imgSrc = `../static/images/${direction}/${source}`

  switch (direction) {
    case 'down':
      trackId = 2 + Math.floor(Math.random() * 2)
      transform = `translateY(${-100 - mapOffsetY}px)`
      break
    case 'right':
      trackId = 4 + Math.floor(Math.random() * 2)
      transform = `translateX(${-100 - mapOffsetX}px)`
      break
    case 'up':
      trackId = 0 + Math.floor(Math.random() * 2)
      transform = `translateY(${868 + mapOffsetY}px)`
      break
    case 'left':
      trackId = 6 + Math.floor(Math.random() * 2)
      transform = `translateX(${868 + mapOffsetX}px)`
      break
    default:
      console.error('Error choosing direction in createVehicle()')
      break
  }

  trackElem = document.querySelector(`#t${trackId}`)

  var img = document.createElement('img')
  img.src = imgSrc

  img.className = `vehicle ${direction}`
  img.id = `vehicle${vehicleId++}`
  img.style.transform = transform
  trackElem.appendChild(img)

  let duration = 15000 + vehicleType * 0.2 * 2000
  animateVehicle(img.id, vehicleType, direction, prevVehicle[trackId], duration, trafficPassed)
  prevVehicle[trackId] = img.id

  if (mode==1){
    let detected = false
    await fetch('/ambulance')
    .then(async (res) => await res.json().then((data) => {
      console.log(data)
      setTimeout(() => {
        changeDetectionStatus(data.emergency_vehicle_detected, data.time_to_be_added, 'green')
      }, 3000);
    }))
  }
}

function changeDetectionStatus(status, timeToBeAdded, color){
  let sts = document.querySelector(".info .container #status")
  let timeToBeAddded = document.querySelector(".info .container #time-to-be-added")
  sts.innerHTML = status
  sts.style.color = color
  setTimeout(() => {
    timeToBeAddded.innerHTML = timeToBeAdded
    timeToBeAddded.style.color = color
  }, 5000);
}

function slowDown(id, vehicleType, direction, prevVehicleId, remainingTime, trafficPassed) {
  animateVehicle(id, vehicleType, direction, prevVehicleId, remainingTime > 12500 ? 12500 : remainingTime + 10, trafficPassed)
}

async function switchSignals(currSignalIndex, nextSignalIndex) {
  let currSC = document.getElementById(`tsc${currSignalIndex}`)
  let nextSC = document.getElementById(`tsc${nextSignalIndex}`)

  var greenSC = 10
  var yellowSC = 5
  var readySC = 10

  enableGreen(currSignalIndex)
  currSC.innerHTML = greenSC--

  function startGreenInterval() {
    enableGreen(currSignalIndex)
    let greenInterval = setInterval(() => {
      if (greenSC >= 0) {
        currSC.innerHTML = greenSC--

        if (greenSC < 3) {
          nextSC.innerHTML = readySC--
        }
      } else {
        currSC.innerHTML = '--'
        disableGreen(currSignalIndex)
        clearInterval(greenInterval)
        startYellowInterval()
        continueReadyInterval()
      }
    }, 1000)
  }

  function startYellowInterval() {
    enableYellow(currSignalIndex)
    let yellowInterval = setInterval(() => {
      if (yellowSC >= 0) {
        currSC.innerHTML = yellowSC--
      } else {
        currSC.innerHTML = '--'
        disableYellow(currSignalIndex)
        enableRed(currSignalIndex)
        clearInterval(yellowInterval)
      }
    }, 1000)
  }

  async function continueReadyInterval() {
    nextSC.innerHTML = readySC--
    let readyInterval = setInterval(() => {
      if (readySC >= 0) {
        nextSC.innerHTML = readySC--
      } else {
        nextSC.innerHTML = '--'
        clearInterval(readyInterval)
        switchSignals(nextSignalIndex, (nextSignalIndex + 1) % 4) // Assuming 4 signals in total
      }
    }, 1000)
  }

  function disableRed(i) {
    signal = document.getElementById(`ts${i}`)
    redLight = signal.querySelector('.frame .light.red')
    redLight.style.backgroundColor = '#7e7e7e'
    signalState[i] = -1
  }

  function disableYellow(i) {
    signal = document.getElementById(`ts${i}`)
    yellowLight = signal.querySelector('.frame .light.yellow')
    yellowLight.style.backgroundColor = '#7e7e7e'
    signalState[i] = -1
  }

  function disableGreen(i) {
    signal = document.getElementById(`ts${i}`)
    greenLight = signal.querySelector('.frame .light.green')
    greenLight.style.backgroundColor = '#7e7e7e'
    signalState[i] = -1
  }

  function enableRed(i) {
    signal = document.getElementById(`ts${i}`)
    disableYellow(i)
    redLight = signal.querySelector('.frame .light.red')
    redLight.style.backgroundColor = 'red'
    signalState[i] = 0
  }

  function enableYellow(i) {
    signal = document.getElementById(`ts${i}`)
    disableGreen(i)
    yellowLight = signal.querySelector('.frame .light.yellow')
    yellowLight.style.backgroundColor = 'yellow'
    signalState[i] = 2
  }

  function enableGreen(i) {
    signal = document.getElementById(`ts${i}`)
    disableRed(i)
    greenLight = signal.querySelector('.frame .light.green')
    greenLight.style.backgroundColor = '#39d600'
    signalState[i] = 1
  }

  startGreenInterval()
}

switchSignals(0, 1)

async function checkTraffic(id, vehicleType, direction, prevVehicleId, remainingTime, trafficPassed, signal) {
  let currentSignal = getSignalState(signal)
  if (currentSignal === 1) {
    trafficPassed = true
    animateVehicle(id, vehicleType, direction, prevVehicleId, 9000 + vehicleType * 0.2 * 2000, trafficPassed)
  }
  animateVehicle(id, vehicleType, direction, prevVehicleId, 9000 + vehicleType * 0.2 * 2000, trafficPassed)

  setTimeout(() => {}, 1)
}

function animateVehicle(id, vehicleType, direction, prevVehicleId, duration, trafficPassed) {
  const v = document.querySelector(`#${id}`)
  let initialPosition
  let targetPosition
  let targetTransform
  let animationId

  switch (direction) {
    case 'down':
      targetPosition = 868 + mapOffsetY
      targetTransform = `translateY(${targetPosition}px)`
      break
    case 'right':
      targetPosition = 868 + mapOffsetX
      targetTransform = `translateX(${targetPosition}px)`
      break
    case 'up':
      targetPosition = -100 - mapOffsetY
      targetTransform = `translateY(${targetPosition}px)`
      break
    case 'left':
      targetPosition = -100 - mapOffsetX
      targetTransform = `translateX(${targetPosition}px)`
      break
  }

  if (direction == 'up' || direction == 'down') {
    initialPosition = v.getBoundingClientRect().y
  } else if (direction == 'right' || direction == 'left') {
    initialPosition = v.getBoundingClientRect().x
  }

  const startTime = performance.now()

  function updatePosition(timestamp) {
    const elapsedTime = timestamp - startTime
    const progress = elapsedTime / duration

    if (progress >= 1) {
      v.style.transform = targetTransform
      return
    }

    const newPosition = initialPosition + progress * (targetPosition - initialPosition)

    const prevVehicle = document.querySelector(`#${prevVehicleId}`)

    switch (direction) {
      case 'up':
        if (!trafficPassed && v.getBoundingClientRect().y - mapOffsetY < 448) {
          checkTraffic(id, vehicleType, direction, prevVehicleId, duration + 10, trafficPassed, 0)
          return
        }
        if (trafficPassed && v.getBoundingClientRect().y - mapOffsetY < -30) {
          v.remove()
          if (vehicleType == 4){
          changeDetectionStatus('No Emergency', 'red')
          }
        }
        break

      case 'down':
        if (!trafficPassed && v.getBoundingClientRect().y + v.clientHeight - mapOffsetY > 320) {
          checkTraffic(id, vehicleType, direction, prevVehicleId, duration - elapsedTime + 10, trafficPassed, 2)
          return
        }
        if (trafficPassed && v.getBoundingClientRect().y + v.clientHeight - mapOffsetY > 808) {
          v.remove()
        }
        break

      case 'left':
        if (!trafficPassed && v.getBoundingClientRect().x - v.clientWidth - mapOffsetX < 400) {
          checkTraffic(id, vehicleType, direction, prevVehicleId, duration - elapsedTime + 10, trafficPassed, 3)
          return
        }
        if (trafficPassed && v.getBoundingClientRect().x - mapOffsetX < -30) {
          v.remove()
        }
        break
      case 'right':
        if (!trafficPassed && v.getBoundingClientRect().x + v.clientWidth - mapOffsetX > 320) {
          checkTraffic(id, vehicleType, direction, prevVehicleId, duration - elapsedTime + 10, trafficPassed, 1)
          return
        }
        if (trafficPassed && v.getBoundingClientRect().x + v.clientWidth - mapOffsetX > 808) {
          v.remove()
        }
        break
    }

    if (prevVehicle && document.contains(prevVehicle)) {
      const distance =
        direction == 'up'
          ? v.getBoundingClientRect().y - prevVehicle.getBoundingClientRect().y - prevVehicle.clientHeight
          : direction == 'down'
          ? prevVehicle.getBoundingClientRect().y - v.clientHeight - v.getBoundingClientRect().y
          : direction == 'left'
          ? v.getBoundingClientRect().x - prevVehicle.getBoundingClientRect().x - prevVehicle.clientWidth
          : direction == 'right'
          ? prevVehicle.getBoundingClientRect().x - v.getBoundingClientRect().x - v.clientWidth
          : null

      if (distance < 10) {
        if (trafficPassed) {
          slowDown(id, vehicleType, direction, prevVehicleId, duration - elapsedTime + 10, trafficPassed)
        } else {
          slowDown(id, vehicleType, direction, prevVehicleId, duration + 10, trafficPassed)
        }
        return
      }
    }

    if (direction == 'up' || direction == 'down') {
      v.style.transform = `translateY(${newPosition}px)`
    } else if (direction == 'right' || direction == 'left') {
      v.style.transform = `translateX(${newPosition}px)`
    }

    animationId = requestAnimationFrame(updatePosition)
  }

  animationId = requestAnimationFrame(updatePosition)
}

setTimeout(() => {
createVehicle(1, 'up')
}, 1500);
createVehicle(0)
createVehicle(0)
createVehicle(0)
createVehicle(0)

createVehicle(0)
setInterval(() => {
  createVehicle(0)
}, 6000)

createVehicle(0)
setInterval(() => {
  createVehicle(0)
}, 15000)
