import chroma from 'chroma-js';
import { motion } from 'framer-motion';
import { FC, useEffect, useState } from 'react';
import Select from 'react-select';
import { AnnotateBlendTag, TextAnnotateBlend } from 'react-text-annotate-blend';

interface SpanInputProps {
  elementId: string;
  text: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string) => void;
  lastTag?: string;
}

export const TextSpanPanel: FC<SpanInputProps> = ({
  elementId,
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
    <div>
      <div className="my-3 w-50 mx-auto d-flex align-items-center">
        <label className="me-2">Annotate with </label>
        <Select
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
          className="btn btn-primary ms-2"
          onClick={() => {
            postAnnotation(JSON.stringify(value) || JSON.stringify([]), elementId);
            setValue([]);
          }}
        >
          Validate annotations
        </button>
      </div>
      <div>
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
    </div>
  );
};
