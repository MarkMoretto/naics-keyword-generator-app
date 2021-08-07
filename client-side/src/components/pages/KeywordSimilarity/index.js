
import { useEffect, useRef, useState } from "react"

// import Griddy from "../../Griddy"

import "./styles.css"


let baseUrl = new URL("http://127.0.0.1:8000/similarity")
const inputPlaceholder = "Enter words here..."

const KeywordSimilarity = () => {

    const [similarKeywords, setSimilarKeywords] = useState("")
    const [postData, setPostData] = useState({
        text_input: "",
        num_results: 10,
    })
    const refTextInput = useRef()

    const onChange = e => {
        setPostData(prevState => {
            return {
                ...prevState,
                [e.target.name]: e.target.value,
            }
        })
    }

    useEffect(() => {
        console.log("useEffect called.")
        // updateQuery(qryParams)
        // refTextInput.current.focus()
    })

    const resetPage = () => {
        // setStopwords({})
        setPostData({
            text_input: "",
            num_results: 10,
        })
        refTextInput.current.focus()
    }

    // health care
    const handleSubmit = e => {
        e.preventDefault()

        fetch(baseUrl, {
            method: "POST",
            headers: {
                "Accept": "application/json",
                "Content-Type":"application/json"
            },
            body: JSON.stringify(postData)
        })
        .then(response => response.json())
        .then((data) => {
            console.log(data)
            setSimilarKeywords(JSON.stringify(data))
        })
        .catch(console.error)
    }

    return (
        <>
            <h2>Relevant Keyword Generator</h2>
            <h5>Enter terms into the form and press enter to retrieve a list of relevant keywords.</h5>
            <div>
                <form onSubmit={handleSubmit} onReset={resetPage} className="form-container">
                    <label htmlFor="id_sample_size">Terms:&nbsp;</label>
                    <input 
                        id="id_sample_size"
                        name="text_input"
                        type="text"
                        placeholder={inputPlaceholder}
                        value={postData.text_input}
                        onChange={onChange}
                        ref={refTextInput}
                        onFocus={(e) => e.target.placeholder = ""} 
                        onBlur={(e) => e.target.placeholder = `${inputPlaceholder}`}
                    />
                    <button className="btn" type="submit">Go!</button>
                    <button className="btn" type="reset">Reset</button>
                </form>
            </div>

            <div className="keyword-list">
                <ol>
                    {similarKeywords ? Object.entries(JSON.parse(similarKeywords)).map(el => {
                        return (
                            <li key={el[0]}>{el[0]}</li>
                        )
                    }): ""}
                </ol>
            </div>
        </>
    )
}

export default KeywordSimilarity