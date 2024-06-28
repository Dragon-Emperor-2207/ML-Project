import React from 'react'
import { useParams } from 'react-router-dom';

const Welcome = (props) => {
  // const {loginId} = useParams();
  // console.log(loginId)
  return (
    <div>
      <button  className='buttonn' type="button" onClick={props.nclose}>|||</button>
      <h1>Welcome</h1>
    </div>
  )
}

export default Welcome