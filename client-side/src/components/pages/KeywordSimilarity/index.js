
import { useEffect, useRef, useState } from "react"
import CheckBox from "../../CheckBox"

import "./styles.css"

/**
 * Constants
*/
const baseUrl = new URL("http://127.0.0.1:8000/similarity")
const inputPlaceholder = "Enter words here..."
const listHeaders = ["rank", "keyword", "score", "irrelevant?"]




/**
 * Round mantissa to a given number of values.
*/
const roundFloat = (obj, numberOfPlaces = 4) => Number.parseFloat(obj).toFixed(numberOfPlaces)



/**
 * KeywordSimilarity component
*/
const KeywordSimilarity = () => {

    const [similarKeywords, setSimilarKeywords] = useState("")
    const [postData, setPostData] = useState({
        text_input: "",
        num_results: 10,
        negative_terms: "",
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
    })

    const resetPage = () => {
        // setStopwords({})
        setPostData({
            text_input: "",
            num_results: 10,
            negative_terms: "",
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

            <div className="item-list">
                <div className="list-row">
                {similarKeywords ? Object.values(listHeaders).map((o, i) => {
                    return (
                        <div key={`header-${i}`} className="list-item centered big-words">{o}</div>
                    )
                }): ""}
                </div>
                {similarKeywords ? Object.entries(JSON.parse(similarKeywords)).map((o, idx) => {
                    return (
                        <div className="list-row">
                            <div className="list-item centered med-words" key={`index-${idx}`}>{idx+1}</div>
                            <div key={`keyword-${idx}`} className="list-item med-words">{o[0]}</div>
                            <div key={`score-${idx}`} className="list-item centered med-words">{roundFloat(o[1])}</div>
                            <div key={`relevant-${idx}`} className="list-item centered med-words">
                                <CheckBox 
                                    key={`checkbox-${idx}`}
                                    name={o[0]}
                                    value={o[0]}
                                />
                            </div>
                        </div>
                    )
                }): ""}


                {/* <ol>
                    {similarKeywords ? Object.entries(JSON.parse(similarKeywords)).map(el => {
                        return (
                            <li key={el[0]}>{el[0]}</li>
                        )
                    }): ""}
                </ol> */}
            </div>
        </>
    )
}

export default KeywordSimilarity