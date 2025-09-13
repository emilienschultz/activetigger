import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';

import { useAddFeature, useDeleteFeature, useGetFeatureInfo } from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { FeatureModelExtended } from '../types';

interface FastTextOptions {
  models?: string[];
}

interface FeaturesOptions {
  fasttext?: FastTextOptions;
}

interface Features {
  options?: FeaturesOptions;
}

export const FeaturesManagement: FC = () => {
  const { projectName } = useParams();

  // get element from the state
  const {
    appContext: { currentProject: project },
  } = useAppContext();

  // API calls
  const { featuresInfo } = useGetFeatureInfo(projectName || null, project);
  const addFeature = useAddFeature();
  const deleteFeature = useDeleteFeature(projectName || null);

  // hooks to use the objets
  const { register, handleSubmit, watch, reset } = useForm<FeatureModelExtended>({
    defaultValues: {
      parameters: {
        dfm_max_term_freq: 100,
        dfm_min_term_freq: 5,
        dfm_ngrams: 1,
        model: 'generic',
      },
      type: 'sbert',
    },
  });

  const { notify } = useNotifications();

  // state for the type of feature to create
  const selectedFeatureToCreate = watch('type');

  // show the menu
  const [showMenu, setShowMenu] = useState(false);

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
    setShowMenu(!showMenu);
  };

  const deleteSelectedFeature = async (element: string) => {
    await deleteFeature(element);
  };

  if (!project) {
    return <div>No project selected</div>;
  }

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="explanations">Available features</div>
        {featuresInfo &&
          Object.entries(featuresInfo).map(([key, value]) => (
            <div className="card text-bg-light mt-3" key={key}>
              <div className="d-flex m-2 align-items-center">
                <button
                  className="btn btn p-0 mx-4"
                  onClick={() => {
                    deleteSelectedFeature(key);
                  }}
                >
                  <MdOutlineDeleteOutline size={20} />
                </button>
                <span className="w-25">{key}</span>
                <span className="mx-2">{value?.time}</span>
                <span className="mx-2">by {value?.user}</span>
                {value?.kind === 'regex' && <span>N={value.parameters['count'] as string}</span>}
              </div>
            </div>
          ))}{' '}
        {/* Display computing features */}
        {Object.entries(project?.features.training).map(([key, element]) => (
          <div className="card text-bg-light mt-3 bg-warning" key={key}>
            <div className="d-flex m-2 align-items-center">
              <span className="w-25">
                Currently computing {element ? element.name : ''}
                {element?.progress ? ` (${element.progress}%)` : ''}
              </span>
            </div>
          </div>
        ))}
        {/* // create new feature */}
        <button onClick={() => setShowMenu(!showMenu)} className="btn btn-primary w-25 mt-3">
          Create a new feature
        </button>
        {showMenu && (
          <form onSubmit={handleSubmit(createNewFeature)} className="mt-3">
            <select className="w-50" id="newFeature" {...register('type')}>
              <option key="empty"></option>
              {Object.keys(project.features.options).map((element) => (
                <option key={element} value={element}>
                  {element}
                </option>
              ))}{' '}
            </select>

            {selectedFeatureToCreate === 'sbert' && (
              <details>
                <summary>Advanced settings</summary>
                <label htmlFor="model">Model to use</label>
                <select id="model" {...register('parameters.model')} className="w-50">
                  <option key={null} value="generic">
                    Default model
                  </option>
                  {(
                    (project?.features.options['sbert']
                      ? (project?.features.options['sbert']['models'] as string[])
                      : []) || []
                  ).map((element) => (
                    <option key={element as string} value={element as string}>
                      {element as string}
                    </option>
                  ))}
                </select>
              </details>
            )}

            {selectedFeatureToCreate === 'fasttext' && (
              <details className="m-2">
                <summary>Advanced settings</summary>
                <label htmlFor="model">Model to use</label>
                <select id="dataset_col" {...register('parameters.model')}>
                  <option key={null} value="generic">
                    Default model
                  </option>

                  {((project?.features as Features).options?.fasttext?.models || []).map(
                    (element) => (
                      <option key={element as string} value={element as string}>
                        {element as string}
                      </option>
                    ),
                  )}
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
              <details className="m-2">
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
                  {(project?.params.all_columns || []).map((element) => (
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
        )}
      </div>
    </div>
  );
};
