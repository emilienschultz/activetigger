import chroma from 'chroma-js';
import { motion } from 'framer-motion';
import { FC, useEffect, useState } from 'react';
import { FaCheck } from 'react-icons/fa';
import { AnnotateBlendTag, TextAnnotateBlend } from 'react-text-annotate-blend';
import { DisplayConfig } from '../../types';

interface SpanInputProps {
  elementId: string;
  displayConfig: DisplayConfig;
  text: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string) => void;
  lastTag?: string;
}

export const TextSpanPanel: FC<SpanInputProps> = ({
  elementId,
  displayConfig,
  text,
  postAnnotation,
  labels,
  lastTag,
}) => {
  // get the context and set the labels

  const [value, setValue] = useState<AnnotateBlendTag[]>([]);
  const [tag, setTag] = useState<string | null>(labels[0] || null);

  useEffect(() => {
    if (lastTag) {
      setValue(JSON.parse(lastTag));
    } else {
      setValue([]);
    }
  }, [lastTag]);

  const handleChange = (value: AnnotateBlendTag[]) => {
    setValue(value);
  };

  const colormap = chroma.scale('Paired').colors(labels.length);
  const COLORS = Object.fromEntries(labels.map((label, index) => [label, colormap[index]]));
  const options = labels.map((label) => ({
    value: label,
    label: label,
    color: COLORS[label],
  }));

  return (
    <>
      <div className="annotation-frame" style={{ height: `${displayConfig.textFrameHeight}vh` }}>
        <motion.div
          animate={elementId ? { backgroundColor: ['#e8e9ff', '#f9f9f9'] } : {}}
          transition={{ duration: 1 }}
        >
          <TextAnnotateBlend
            style={{
              fontSize: '1.2rem',
            }}
            content={text || ''}
            onChange={handleChange}
            value={value || []}
            getSpan={(span) => {
              if (!tag) return span;
              return {
                ...span,
                tag: tag,
                color: tag && COLORS[tag],
              };
            }}
          />
        </motion.div>
      </div>
      <div>
        <div className="d-flex gap-2 align-items-center mt-2">
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
              width: '100%',
            }}
          >
            {options.map((opt) => {
              const isActive = opt.value === tag;

              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setTag(opt.value)}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: `1px solid ${opt.color}`,
                    backgroundColor: isActive ? opt.color : 'white',
                    color: isActive ? 'white' : opt.color,
                    cursor: 'pointer',
                    fontWeight: isActive ? 600 : 500,
                    transition: 'background-color 0.15s ease, color 0.15s ease',
                  }}
                >
                  {opt.label}
                </button>
              );
            })}
            <button
              className="btn btn-outline-success align-items-center justify-content-center validate-btn"
              onClick={() => {
                postAnnotation(JSON.stringify(value) || JSON.stringify([]), elementId);
                setValue([]);
              }}
              style={{
                width: '100%',
                textAlign: 'center',
                padding: '2px 2px',
                borderRadius: '6px',
                border: `1px solid green`,
                backgroundColor: 'white',
                color: 'green',
                cursor: 'pointer',
                transition: 'background-color 0.15s ease, color 0.15s ease',
              }}
            >
              <FaCheck size={18} /> Validate the annotation
            </button>
          </div>
          {/* <Select
            value={options.find((opt) => opt.value === tag) || null}
            onChange={(opt) => setTag(opt && opt.value)}
            options={options}
            styles={{
              option: (provided, state) => ({
                ...provided,
                backgroundColor: state.isFocused ? state.data.color : 'white',
                color: state.isFocused ? 'white' : state.data.color,
              }),
              singleValue: (provided, state) => ({
                ...provided,
                color: state.data.color,
              }),
            }}
          />

          <button
            className="btn btn-outline-success align-items-center justify-content-center validate-btn"
            onClick={() => {
              postAnnotation(JSON.stringify(value) || JSON.stringify([]), elementId);
              setValue([]);
            }}
          >
            <FaCheck size={18} />
          </button> */}
        </div>
      </div>
    </>
  );
};
