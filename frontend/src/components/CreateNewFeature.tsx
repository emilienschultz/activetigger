import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useParams } from 'react-router-dom';

import { useAddFeature } from '../core/api';
import { useNotifications } from '../core/notifications';
import { FeatureModelExtended } from '../types';

interface Options {
  models?: string[];
}

interface FeaturesOptions {
  fasttext?: Options;
  sbert?: Options;
}

interface CreateNewFeatureProps {
  projectName?: string;
  columns: string[];
  featuresOption: FeaturesOptions;
  callback?: (state: boolean) => void;
}

export const CreateNewFeature: FC<CreateNewFeatureProps> = ({
  featuresOption,
  columns,
  callback,
}) => {
  const { projectName } = useParams();

  // API calls
  const addFeature = useAddFeature();

  // hooks to use the objets
  const { register, handleSubmit, watch, reset } = useForm<FeatureModelExtended>({
    defaultValues: {
      parameters: {
        dfm_max_term_freq: 100,
        dfm_min_term_freq: 5,
        dfm_ngrams: 1,
        model: 'generic',
        max_length_tokens: 1024,
      },
      type: 'sbert',
    },
  });

  const { notify } = useNotifications();

  // state for the type of feature to create
  const selectedFeatureToCreate = watch('type');

  // action to create the new feature
  const createNewFeature: SubmitHandler<FeatureModelExtended> = async (formData) => {
    try {
      addFeature(
        projectName || null,
        formData.type,
        formData.name,
        formData.parameters as unknown as Record<string, string | number | undefined>,
      );
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    reset();
    if (callback) callback(false);
  };

  return (
    <div>
      <form onSubmit={handleSubmit(createNewFeature)} className="mt-3">
        <label htmlFor="newFeature">Feature type</label>
        <select id="newFeature" {...register('type')}>
          <option key="empty"></option>
          {Object.keys(featuresOption).map((element) => (
            <option key={element} value={element}>
              {element}
            </option>
          ))}{' '}
        </select>

        {selectedFeatureToCreate === 'sbert' && (
          <details className="custom-details">
            <summary>Advanced settings</summary>
            <label htmlFor="model">Model to use</label>
            <select id="model" {...register('parameters.model')}>
              <option key={null} value="generic">
                Default model
              </option>
              {(
                (featuresOption.sbert ? (featuresOption['sbert']['models'] as string[]) : []) || []
              ).map((element) => (
                <option key={element as string} value={element as string}>
                  {element as string}
                </option>
              ))}
            </select>
            <label htmlFor="length">Max length tokens</label>
            <input
              type="number"
              className="form-control"
              placeholder="Max length tokens"
              {...register('parameters.max_length_tokens')}
            />
          </details>
        )}

        {selectedFeatureToCreate === 'fasttext' && (
          <details className="custom-details">
            <summary>Advanced settings</summary>
            <label htmlFor="model">Model to use</label>
            <select id="dataset_col" {...register('parameters.model')}>
              <option key={null} value="generic">
                Default model
              </option>

              {(featuresOption.fasttext?.models || []).map((element) => (
                <option key={element as string} value={element as string}>
                  {element as string}
                </option>
              ))}
            </select>
          </details>
        )}

        {selectedFeatureToCreate === 'regex' && (
          <input
            type="text"
            className="form-control mt-3"
            placeholder="Enter the regex"
            {...register('parameters.value')}
          />
        )}

        {selectedFeatureToCreate === 'dfm' && (
          <details className="custom-details">
            <summary>Advanced settings</summary>
            <div>
              <label htmlFor="dfm_tfidf">TF-IDF</label>
              <select id="dfm_tfidf" {...register('parameters.dfm_tfidf')}>
                <option key="true">True</option>
                <option key="false">False</option>
              </select>
            </div>
            <div>
              <label htmlFor="dfm_ngrams">Ngrams</label>
              <input type="number" id="dfm_ngrams" {...register('parameters.dfm_ngrams')} />
            </div>
            <div>
              <label htmlFor="dfm_min_term_freq">Min term freq</label>
              <input
                type="number"
                id="dfm_min_term_freq"
                {...register('parameters.dfm_min_term_freq')}
              />
            </div>
            <div>
              <label htmlFor="dfm_max_term_freq">Max term freq</label>
              <input
                type="number"
                id="dfm_max_term_freq"
                {...register('parameters.dfm_max_term_freq')}
              />
            </div>
            <div>
              <label htmlFor="dfm_norm">Norm</label>
              <select id="dfm_norm" {...register('parameters.dfm_norm')}>
                <option key="true">True</option>
                <option key="false">False</option>
              </select>
            </div>
            <div>
              <label htmlFor="dfm_log">Log</label>
              <select id="dfm_log" {...register('parameters.dfm_log')}>
                <option key="true">True</option>
                <option key="false">False</option>
              </select>
            </div>
          </details>
        )}

        {selectedFeatureToCreate === 'dataset' && (
          <div>
            <label htmlFor="dataset_col">Column to use</label>
            <select id="dataset_col" {...register('parameters.dataset_col')}>
              {columns.map((element) => (
                <option key={element as string} value={element as string}>
                  {element as string}
                </option>
              ))}
            </select>
            <label htmlFor="dataset_type">Type of the feature</label>
            <select id="dataset_type" {...register('parameters.dataset_type')}>
              <option key="numeric">Numeric</option>
              <option key="categorical">Categorical</option>
            </select>
          </div>
        )}
        <button className="btn btn-primary w-25">Compute</button>
      </form>
    </div>
  );
};
