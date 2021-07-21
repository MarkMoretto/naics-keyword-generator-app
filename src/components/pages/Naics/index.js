
import { useState } from "react"

import { naicsUrl } from "./utils"

const Naics = () => {

    console.log(useState("Naics Component."))

    return (
        <>
            <h1>Oh, baby.</h1>
            <h2>Oh, baby.</h2>
            <h3>Oh, baby.</h3>
            <h4>Oh, baby.</h4>
            <h5>Oh, baby.</h5>
            <h6>Oh, baby.</h6>
            <div>
                <p>Here is a URL: <span className="stronger">{naicsUrl}</span></p>
                <a href={naicsUrl} target="_blank" rel="noreferrer">Here</a> is a link.
            </div>
        </>
    )
}

export default Naics