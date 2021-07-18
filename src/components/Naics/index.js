
import React, { useState } from "react"

import "./styles.css"

const Naics = () => {

    const naics_url = "https://www.census.gov/naics/?58967?yearbck=2017"

    console.log(useState("Naics Component."))

    return (
        <React.Fragment>
            <h1>Oh, baby.</h1>
            <h2>Oh, baby.</h2>
            <h3>Oh, baby.</h3>
            <h4>Oh, baby.</h4>
            <h5>Oh, baby.</h5>
            <h6>Oh, baby.</h6>
            <div>
                <p>Here is a URL: <span className="stronger">{naics_url}</span></p>
                <a href={naics_url} target="_blank" rel="noreferrer">Here</a> is a link.
            </div>
        </React.Fragment>
    )
}

export default Naics
